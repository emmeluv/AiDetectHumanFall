import streamlit as st
import cv2
import cvzone
import math
from ultralytics import YOLO
import tempfile

# โหลดโมเดล YOLOv8 ที่ผ่านการเทรน
model = YOLO("best.pt")

# ฟังก์ชันสำหรับประมวลผลเฟรม
def process_frame(frame):
    classnames = []
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()

    # ประมวลผลเฟรมด้วยโมเดล YOLOv8
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 70 and class_detect == 'Person':
                cvzone.cornerRect(frame, (x1, y1, width, height), l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect} {conf}%', (x1 + 8, y1 - 12), thickness=2, scale=2)

            if threshold < 5:
                cvzone.cornerRect(frame, (x1, y1, width, height), l=30, rt=6)
                cvzone.putTextRect(frame, 'Fall Detected', (x1, y1), thickness=2, scale=2)

    return frame

# สร้างแอป Streamlit
st.title('Human Fall Detection')

# ตัวเลือกให้เลือกใช้กล้องหรืออัปโหลดวิดีโอ
option = st.selectbox("Choose an option:", ["Use Webcam", "Upload Video"])

# Placeholder สำหรับแสดงเฟรม
frame_placeholder = st.empty()

if option == "Use Webcam":
    # Toggle สำหรับเปิด/ปิดกล้อง
    camera_running = st.toggle('Start Webcam', value=False)

    if camera_running:
        # เปิดกล้องผ่าน OpenCV
        cap = cv2.VideoCapture(0)  # 0 คือกล้องหลัก

        # วนลูปเพื่ออ่านเฟรมจากกล้อง
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read from webcam.")
                break

            frame = cv2.resize(frame, (980, 740))

            # ประมวลผลเฟรม
            processed_frame = process_frame(frame)

            # แสดงเฟรมที่ประมวลผลแล้วใน Streamlit
            frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

        # ปล่อยกล้องเมื่อเสร็จสิ้น
        cap.release()
    else:
        frame_placeholder.image([])  # ล้างภาพเมื่อปิดกล้อง

elif option == "Upload Video":
    # อัปโหลดไฟล์วิดีโอ
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # สร้างไฟล์ชั่วคราวสำหรับบันทึกวิดีโอที่อัปโหลด
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        # เปิดวิดีโอที่อัปโหลด
        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            st.error("Error: Could not open video.")
        else:
            # วนลูปเพื่ออ่านเฟรมจากวิดีโอที่อัปโหลด
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (980, 740))

                # ประมวลผลเฟรม
                processed_frame = process_frame(frame)

                # แสดงเฟรมที่ประมวลผลแล้วใน Streamlit
                frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

        # ปล่อยวิดีโอเมื่อเสร็จสิ้น
        cap.release()
