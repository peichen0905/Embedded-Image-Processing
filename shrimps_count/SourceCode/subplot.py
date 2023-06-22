import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from statistics import mode

class Shrimp:
    def __init__(self):
        self.contours = []
        self.frame_count = None
        self.gray_frames = []
        self.gray_median = None
        self.gray = None
        self.frame = None
        self.gray_diff = None
        self.division_results = []

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("无法打开视频文件")
            return

        self.frame_count = 0
        self.gray_frames = []

        while True:
            ret, frame1 = cap.read()

            if not ret:
                break

            self.frame = cv2.resize(frame1, (1920, 1080))

            self.TemporalFiltering()
            self.BackgroundModeling()

            if cv2.waitKey(1) == 27:
                break

        self.shrimp_num()
        cap.release()
        cv2.destroyAllWindows()

    def TemporalFiltering(self):
        self.frame_count += 1
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.gray_frames.append(self.gray)

        if self.frame_count % 200 == 0:
            gray_frames = np.array(self.gray_frames)
            self.gray_median = np.median(gray_frames, axis=0).astype(np.uint8)
            self.gray_frames = []

    def BackgroundModeling(self):
        if self.gray_median is not None:
            self.gray_diff = cv2.absdiff(self.gray, self.gray_median)
            _, self.thresh = cv2.threshold(self.gray_diff, 90, 255, cv2.THRESH_BINARY)
            cv2.imshow("frame", self.frame)
            cv2.imshow("diff", self.thresh)

            contour_count, contour_areas = self.calculate_contour_areas(self.thresh)
            self.contours = contour_count

            contour_areas_array = np.array(contour_areas)
            contour_area_sum = np.sum(contour_areas_array * 10)
            contour_area_mode = int(np.median(contour_areas_array * 10))

            if contour_area_mode == 0:
                contour_area_mode = 1

            contour_area_division = contour_area_sum / contour_area_mode

            self.division_results.append(contour_area_division)

            print("area_sum:", contour_area_sum)
            print("area_mode:", contour_area_mode)
            print("division:", contour_area_division)

    def shrimp_num(self):
        division_results_array = np.array(self.division_results)
        division_results_array = division_results_array[division_results_array != 0]  # 去除值为0的元素
        division_results_array = division_results_array.astype(int)

        division_results_mode = int(np.bincount(division_results_array).argmax())
        if division_results_mode == 0:
            division_results_mode = 1

        print("蝦苗數量:", division_results_mode)
        y = np.bincount(division_results_array)
        x = np.arange(len(y))
        plt.plot(x, y, color='green', linewidth=1)

        max_index = np.argmax(y)
        max_value = y[max_index]
        plt.annotate(f"Max: {max_value} (x={max_index})", xy=(max_index, max_value),
                     xytext=(max_index + 30, max_value - 0.04 * max_value),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

        plt.xlabel('Division Results')
        plt.ylabel('Frequency')
        plt.title('Distribution of Division Results')
        plt.show()

    def calculate_contour_areas(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        contour_areas = [cv2.contourArea(contour) for contour in contours]
        return contour_count, contour_areas


shrimp = Shrimp()
video_path = "Shrimp2.mp4"
start_time = time.time()
shrimp.process_video(video_path)
end_time = time.time()
print("time:", end_time - start_time)