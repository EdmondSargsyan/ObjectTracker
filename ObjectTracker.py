import cv2

cap = cv2.VideoCapture('cars.mp4')

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}

trackers = cv2.legacy.MultiTracker_create()

while True:
    frame = cap.read()[1]
    if frame is None:
        break
    frame = cv2.resize(frame, (1080, 600))

    (success, boxes) = trackers.update(frame)
    print(success, boxes)
    if success == False:
        bound_boxes = trackers.getObjects()
        # idx = np.where(bound_boxes.sum(axis=5) != 0)[0]
        # bound_boxes = bound_boxes[idx]
        for bound_box in bound_boxes:
            trackers.add(tracker, frame, bound_box)

    for i, box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 10, 255), 2)
        cv2.putText(frame, 'TRACKING', (x + 20, y - 3), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 0), 2)

    cv2.imshow('Following', frame)
    k = cv2.waitKey(40)

    if k == ord(" "):
        Roi = cv2.selectROI('Following', frame, fromCenter=False,
                            showCrosshair=True)

        tracker = OPENCV_OBJECT_TRACKERS['kcf']()
        trackers.add(tracker, frame, Roi)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
