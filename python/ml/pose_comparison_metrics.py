# This is code that I wrote for Deep Object Pose, but I do not want to be
# constrained to a non-commercial license.


def load_model_cm(filename):
    assert filename.endswith(".obj")
    v = []
    with open(filename) as f:
        for line in f.readlines():
            if not line.startswith("v "):
                continue
            v.append([float(x) for x in line[2:].split(" ")])
    return {"pts": np.asarray(v)}


class Pose(object):
    # Intended to show pose of model in camera frame.
    def __init__(self, R, t_cm):
        self.R = R
        self.t_cm = t_cm


class Comparison(object):
    def __init__(self, pose_est_list, pose_gt_list):
        self.pose_est_list = pose_est_list
        self.pose_gt_list = pose_gt_list
        self.num_est = len(pose_est_list)
        self.num_gt = len(pose_gt_list)
        self.est_to_gt_indices = None

    def greedy_match(self, pose_error_func):
        # Greedy match.
        self.est_to_gt_indices = np.array([-1] * self.num_est, dtype=int)
        matched_gt = np.zeros(self.num_gt, dtype=bool)
        for i_est, pose_est in enumerate(self.pose_est_list):
            if np.all(matched_gt):
                break
            dist_to_gt = np.array([
                pose_error_func(pose_est, pose_gt)
                for pose_gt in self.pose_gt_list])
            dist_to_gt[matched_gt] = np.inf
            i_gt = np.argmin(dist_to_gt)
            self.est_to_gt_indices[i_gt] = i_gt
            matched_gt[i_gt] = True

    dtype = np.dtype([
        ('num_est', float),
        ('num_gt', float),
        ('num_tp', float),
        ('num_fp', float),
        ('num_fn', float),
        ('precision', float),
        ('recall', float),
        ('f1_score', float),
    ])

    def compute_metrics(self, pose_error_func, error_threshold):
        assert self.est_to_gt_indices is not None
        est_accurate = np.zeros(self.num_est, dtype=bool)
        for i_est, pose_est in enumerate(self.pose_est_list):
            i_gt = self.est_to_gt_indices[i_est]
            if i_gt == -1:
                continue
            pose_gt = self.pose_gt_list[i_gt]
            pose_error = pose_error_func(pose_est, pose_gt)
            if pose_error < error_threshold:
                est_accurate[i_est] = True
        num_tp = np.sum(est_accurate)
        num_fp = self.num_est - num_tp
        num_fn = np.sum(self.est_to_gt_indices == -1)
        # TODO(eric): Er... How do I compute true negatives???
        # For now, just gonna do F1 score...
        # Convention:
        # - No detections, precision is 100%.
        # - No labels, recall is 0%.
        if self.num_est == 0:
            precision = 1.
        else:
            precision = num_tp / self.num_est
        if self.num_gt == 0:
            recall = 0.
        else:
            recall = num_tp / self.num_gt
        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        return np.rec.array((
            self.num_est,
            self.num_gt,
            num_tp,
            num_fp,
            num_fn,
            precision,
            recall,
            f1_score,
        ), dtype=self.dtype)

