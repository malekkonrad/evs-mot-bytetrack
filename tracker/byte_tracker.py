"""Tracker w stylu ByteTrack: predefiniowane detekcje + filtr Kalmana +
dwustopniowa asocjacja (high-score, potem low-score).

Kluczowe pomysły z ByteTrack (Zhang et al. 2022):
- Detekcje powyżej track_thresh ("high") trafiają do pierwszej asocjacji
  z aktywnymi i straconymi torami. Niedopasowane high-score detekcje (które
  przekraczają new_track_thresh) inicjalizują nowe tory (najpierw jako
  unconfirmed — potwierdzone dopiero gdy znów zostaną dopasowane).
- Detekcje poniżej track_thresh ("low") ratują częściowo zasłonięte obiekty
  w drugiej asocjacji — dopasowywane są wyłącznie do torów, które przeżyły
  pierwszą rundę i są w stanie Tracked (nie Lost).
- Tory bez dopasowania → Lost; usunięcie po max_time_lost klatkach.
"""

from __future__ import annotations

import numpy as np

from kalman_filter import KalmanFilter
from matching import fuse_score, iou_distance, iou_matrix, linear_assignment
from cmc import affine_to_state8


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


def _tlwh_to_cxywh(tlwh: np.ndarray) -> np.ndarray:
    out = np.asarray(tlwh, dtype=float).copy()
    out[:2] = out[:2] + out[2:] / 2.0
    return out


class Track:
    _next_id = 0

    def __init__(self, tlwh: np.ndarray, score: float):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.score = float(score)
        self.kalman_filter: KalmanFilter | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.track_id = 0
        self.state = TrackState.New
        self.is_activated = False
        self.start_frame = 0
        self.frame_id = 0
        self.tracklet_len = 0

    @staticmethod
    def reset_id() -> None:
        Track._next_id = 0

    @staticmethod
    def next_id() -> int:
        Track._next_id += 1
        return Track._next_id

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=float)

    def predict(self, lost_velocity_damping: float = 1.0) -> None:
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            # Wygaszamy prędkości w/h gdy tor jest stracony — bb nie powinno
            # nadal "rosnąć/maleć" bez nowych obserwacji.
            mean_state[6] = 0.0
            mean_state[7] = 0.0
            # Tłumienie prędkości środka — zapobiega rozjechaniu predykcji
            mean_state[4] *= lost_velocity_damping
            mean_state[5] *= lost_velocity_damping
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks: list["Track"], lost_velocity_damping: float = 1.0) -> None:
        for t in tracks:
            t.predict(lost_velocity_damping=lost_velocity_damping)

    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = Track.next_id()
        self.mean, self.covariance = kalman_filter.initiate(_tlwh_to_cxywh(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1  # tylko pierwsza klatka ⇒ od razu confirmed
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "Track", frame_id: int, new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, _tlwh_to_cxywh(new_track._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        if new_id:
            self.track_id = Track.next_id()

    def update(self, new_track: "Track", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, _tlwh_to_cxywh(new_track._tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


def _joint(a: list[Track], b: list[Track]) -> list[Track]:
    seen = {t.track_id for t in a}
    return list(a) + [t for t in b if t.track_id not in seen]


def _sub(a: list[Track], b: list[Track]) -> list[Track]:
    rm = {t.track_id for t in b}
    return [t for t in a if t.track_id not in rm]


def _remove_duplicate_tracks(a: list[Track], b: list[Track]) -> tuple[list[Track], list[Track]]:
    """Usuwa duplikaty pomiędzy aktywnymi a straconymi torami (IoU > 0.85).

    Zachowuje tor o dłuższej historii.
    """
    if not a or not b:
        return a, b
    boxes_a = np.asarray([t.tlwh for t in a], dtype=float)
    boxes_b = np.asarray([t.tlwh for t in b], dtype=float)
    pdist = 1.0 - iou_matrix(boxes_a, boxes_b)
    pairs = np.where(pdist < 0.15)
    dup_a, dup_b = set(), set()
    for p, q in zip(*pairs):
        time_p = a[p].frame_id - a[p].start_frame
        time_q = b[q].frame_id - b[q].start_frame
        if time_p > time_q:
            dup_b.add(int(q))
        else:
            dup_a.add(int(p))
    res_a = [t for i, t in enumerate(a) if i not in dup_a]
    res_b = [t for i, t in enumerate(b) if i not in dup_b]
    return res_a, res_b


class ByteTracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        new_track_thresh: float = 0.6,
        low_thresh: float = 0.1,
        track_buffer: int = 30,
        frame_rate: float = 30.0,
        use_score_fusion: bool = True,
        predict_lost_frames: int = 0,
        min_box_area: float = 0.0,
        lost_velocity_damping: float = 1.0,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.new_track_thresh = new_track_thresh
        self.low_thresh = low_thresh
        self.use_score_fusion = use_score_fusion
        self.predict_lost_frames = predict_lost_frames
        self.min_box_area = min_box_area
        self.lost_velocity_damping = lost_velocity_damping
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.tracked_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.removed_tracks: list[Track] = []
        Track.reset_id()

    def apply_cmc(self, affine: np.ndarray) -> None:
        """Koryguje stan wszystkich tracków (active + lost) o ruch kamery.

        affine: 2x3, mapuje współrzędne poprzedniej klatki na obecną.
        Pomija identycznościowe transformacje dla efektywności.
        """
        if affine is None:
            return
        if np.allclose(affine, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])):
            return
        M8, T8 = affine_to_state8(affine)
        for t in self.tracked_tracks + self.lost_tracks:
            if t.mean is None:
                continue
            t.mean = M8 @ t.mean + T8
            t.covariance = M8 @ t.covariance @ M8.T

    def update(self, dets_xywh: np.ndarray, scores: np.ndarray) -> list[Track]:
        self.frame_id += 1
        activated: list[Track] = []
        refound: list[Track] = []
        lost_now: list[Track] = []
        removed_now: list[Track] = []

        if len(dets_xywh) == 0:
            dets_xywh = np.empty((0, 4), dtype=float)
            scores = np.empty((0,), dtype=float)
        else:
            dets_xywh = np.asarray(dets_xywh, dtype=float)
            scores = np.asarray(scores, dtype=float)

        high_mask = scores >= self.track_thresh
        low_mask = (scores >= self.low_thresh) & (scores < self.track_thresh)

        dets_high = dets_xywh[high_mask]
        scores_high = scores[high_mask]
        dets_low = dets_xywh[low_mask]
        scores_low = scores[low_mask]

        # Podział aktywnych torów: confirmed vs unconfirmed (świeżo narodzone)
        unconfirmed = [t for t in self.tracked_tracks if not t.is_activated]
        tracked = [t for t in self.tracked_tracks if t.is_activated]

        track_pool = _joint(tracked, self.lost_tracks)
        Track.multi_predict(track_pool, lost_velocity_damping=self.lost_velocity_damping)
        Track.multi_predict(unconfirmed)

        # === ASOCJACJA #1: dets_high vs (tracked + lost) ===
        cost = iou_distance(track_pool, dets_high)
        if self.use_score_fusion and cost.size > 0:
            cost = fuse_score(cost, scores_high)
        matches, u_track, u_det_high = linear_assignment(cost, thresh=self.match_thresh)

        for itr, idet in matches:
            track = track_pool[itr]
            new_t = Track(dets_high[idet], scores_high[idet])
            if track.state == TrackState.Tracked:
                track.update(new_t, self.frame_id)
                activated.append(track)
            else:  # Lost → re-activate
                track.re_activate(new_t, self.frame_id, new_id=False)
                refound.append(track)

        # === ASOCJACJA #2: dets_low vs niedopasowane TRACKED tory (bez lost!) ===
        r_tracked = [track_pool[i] for i in u_track if track_pool[i].state == TrackState.Tracked]
        cost2 = iou_distance(r_tracked, dets_low)
        # bez fuse_score na niskich detekcjach (ich score jest niewiarygodny)
        matches2, u_track2, _ = linear_assignment(cost2, thresh=0.5)

        for itr, idet in matches2:
            track = r_tracked[itr]
            new_t = Track(dets_low[idet], scores_low[idet])
            track.update(new_t, self.frame_id)
            activated.append(track)

        # Tracked, które przeżyły obie rundy bez dopasowania → Lost
        for i in u_track2:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_now.append(track)

        # === ASOCJACJA #3: pozostałe dets_high vs unconfirmed ===
        u_dets_high_idx = u_det_high
        cost3 = iou_distance(unconfirmed, dets_high[u_dets_high_idx])
        if self.use_score_fusion and cost3.size > 0:
            cost3 = fuse_score(cost3, scores_high[u_dets_high_idx])
        matches3, u_unconf, u_det3 = linear_assignment(cost3, thresh=0.7)

        for itr, idet in matches3:
            track = unconfirmed[itr]
            real_det_idx = u_dets_high_idx[idet]
            new_t = Track(dets_high[real_det_idx], scores_high[real_det_idx])
            track.update(new_t, self.frame_id)
            activated.append(track)

        for i in u_unconf:
            track = unconfirmed[i]
            track.mark_removed()
            removed_now.append(track)

        # === Inicjalizacja nowych torów z high-score, niedopasowanych ===
        for i in u_det3:
            real_det_idx = u_dets_high_idx[i]
            score_i = float(scores_high[real_det_idx])
            if score_i < self.new_track_thresh:
                continue
            new_t = Track(dets_high[real_det_idx], score_i)
            new_t.activate(self.kalman_filter, self.frame_id)
            activated.append(new_t)

        # === Czyszczenie buforów ===
        for t in self.lost_tracks:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.mark_removed()
                removed_now.append(t)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = _joint(self.tracked_tracks, activated)
        self.tracked_tracks = _joint(self.tracked_tracks, refound)
        self.lost_tracks = _sub(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_now)
        self.lost_tracks = _sub(self.lost_tracks, removed_now)
        self.removed_tracks.extend(removed_now)

        self.tracked_tracks, self.lost_tracks = _remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )

        output: list[Track] = [t for t in self.tracked_tracks if t.is_activated]
        # Wypełnianie luk detekcji predykcją Kalmana dla niedawno straconych
        # torów. Daje (zwykle) zysk na recall przy minimalnym koszcie precyzji.
        if self.predict_lost_frames > 0:
            for t in self.lost_tracks:
                if not t.is_activated:
                    continue
                gap = self.frame_id - t.frame_id
                if 1 <= gap <= self.predict_lost_frames:
                    output.append(t)
        if self.min_box_area > 0:
            output = [t for t in output if t.tlwh[2] * t.tlwh[3] >= self.min_box_area]
        return output
