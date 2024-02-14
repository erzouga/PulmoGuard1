import cv2
import pydicom as dicom


class NoduleDatLoader:
    @staticmethod
    def dcom_image_reader(image_path):
        ds = dicom.dcmread(image_path)
        pixel_array_numpy = ds.pixel_array
        return pixel_array_numpy

    @staticmethod
    def show_image(image, size=(400, 400)):
        resized_image = cv2.resize(image, size)
        cv2.imshow("image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


image_path = r'C:\Users\erzou\Desktop\lUNG CANCER\manifest-1707553744722\LIDC-IDRI\LIDC-IDRI-0002\01-01-2000-NA-NA-98329\3000522.000000-NA-04919\1-041.dcm'
ndl = NoduleDatLoader()
img = ndl.dcom_image_reader(image_path)
ndl.show_image(img)