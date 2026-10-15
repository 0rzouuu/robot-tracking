import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hsv_values = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = param['hsv']
        h, s, v = hsv[y, x]
        hsv_values.append((h, s, v))
        print(f"Clicked at ({x},{y}) -> H:{h}  S:{s}  V:{v}")
        if len(hsv_values) >= 2:
            hs = [p[0] for p in hsv_values]
            ss = [p[1] for p in hsv_values]
            vs = [p[2] for p in hsv_values]
            print(f"\n--- Suggested ranges from {len(hsv_values)} samples ---")
            print(f"  Hue:  {max(0, min(hs)-10)} - {min(179, max(hs)+10)}")
            print(f"  Sat:  {max(0, min(ss)-40)} - {min(255, max(ss)+40)}")
            print(f"  Val:  {max(0, min(vs)-40)} - {min(255, max(vs)+40)}")
            print()

print("=" * 50)
print("HSV COLOR PICKER")
print("=" * 50)
print("Click on the yellow ball in different spots.")
print("The more clicks, the better the range estimate.")
print("Press Q to quit.")
print("=" * 50)

cv2.namedWindow("HSV Picker")
data = {'hsv': None}
cv2.setMouseCallback("HSV Picker", on_click, data)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    data['hsv'] = hsv

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    roi = hsv[cy-2:cy+2, cx-2:cx+2]
    avg_h = int(np.mean(roi[:,:,0]))
    avg_s = int(np.mean(roi[:,:,1]))
    avg_v = int(np.mean(roi[:,:,2]))

    cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(frame, f"Center: H:{avg_h} S:{avg_s} V:{avg_v}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Click on the ball | Q to quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("HSV Picker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if hsv_values:
    hs = [p[0] for p in hsv_values]
    ss = [p[1] for p in hsv_values]
    vs = [p[2] for p in hsv_values]
    print("\n" + "=" * 50)
    print("FINAL SUGGESTED HSV RANGES")
    print("=" * 50)
    print(f"  Hue Lower:  {max(0, min(hs)-10)}")
    print(f"  Hue Upper:  {min(179, max(hs)+10)}")
    print(f"  Sat Low:    {max(0, min(ss)-40)}")
    print(f"  Sat High:   {min(255, max(ss)+40)}")
    print(f"  Val Low:    {max(0, min(vs)-40)}")
    print(f"  Val High:   {min(255, max(vs)+40)}")
    print("=" * 50)
    print("Copy these values into the HSV sliders in the web UI.")
