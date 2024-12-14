def update_tracked_data(tracked_data, detections):
    for track_id, bbox in detections:
        if track_id not in tracked_data:
            tracked_data[track_id] = []
        tracked_data[track_id].append(bbox)

def predict_next_bbox(tracked_data, track_id, num_frames=2):
    if len(tracked_data[track_id]) < num_frames:
        return None
    recent_bboxes = tracked_data[track_id][-num_frames:]
    deltas = [recent_bboxes[i+1][j] - recent_bboxes[i][j] for i in range(num_frames-1) for j in range(4)]
    avg_delta = [sum(deltas[i::4]) / (num_frames-1) for i in range(4)]
    predicted_bbox = [recent_bboxes[-1][j] + avg_delta[j] for j in range(4)]
    return predicted_bbox
