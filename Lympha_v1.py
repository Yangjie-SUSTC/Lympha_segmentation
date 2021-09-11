import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import base64
import json
from sklearn.cluster import KMeans


class mouse:
    def __init__(self, img_path):
        self.path = img_path
        self.type = img_path.split('\\')[-1][-1]
        self.name = img_path.split('\\')[-1].split('.')[0]
        self.folder = img_path.split(self.name)[0]
        self.read_img()
        self.body_json_path = os.path.join(self.folder, 'equalized', 'labelme', img_name + '.json')
        self.ROI_json_path = os.path.join(self.folder, '8bit', 'labelme', img_name + '.json')
        self.mask_body = None
        self.masked_body = None

    def debug_show_mask(self, mask):
        merg_mask = self.get_merg_mask(self.color_8, mask)
        self.show_img(merg_mask, name='debug')

    def pipline(self):
        self.mask_body, self.masked_body = self.body(all_step=True, debug=False)
        self.result = self.ROI(all_step=True, debug=True)
        self.save_img(self.mask_data, self.mask_body)

        # self.show_img(merg_mask,name='auto ROI')
        # self.show_contours(self.color_8,contours)

    def save_img(self, mask_data, mask_body):
        color = {'le': [0], 'lc': [1], 'rc': [2], 're': [0, 1]}
        label = {'le': 2, 'lc': 3, 'rc': 4, 're':5}
        total = np.zeros_like(self.gray_8)
        merg_mask2 = self.color_8
        lympha = np.zeros_like(self.gray_8)
        for key, value in mask_data.items():
            mask = np.zeros_like(self.gray_8)
            path = os.path.join(self.folder, 'mask', key, self.name + '.png')
            mask[value > 0] = 255
            total[value > 0] = label[key]
            merg_mask2 = self.get_merg_mask(merg_mask2, value, index=color[key])
            cv2.imwrite(path, mask)

        lympha[total > 0] = 255
        name = os.path.join(self.folder, 'mask', 'lympha', self.name + '.png')
        cv2.imwrite(name, lympha)

        mask1 = np.ones_like(self.gray_8)
        mask1[mask_body] = 0
        name = os.path.join(self.folder, 'mask', 'bg', self.name + '.png')
        cv2.imwrite(name, mask1 * 255)

        timg = total + mask1
        path = os.path.join(self.folder, 'mask', 'total', self.name + '.png')
        cv2.imwrite(path, timg)

        contours, hierarchy = cv2.findContours((1 - mask1) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(merg_mask2.copy(), contours, -1, (255, 255, 0), 2)
        path = os.path.join(self.folder, 'result', self.name + '.png')
        cv2.imwrite(path, img)
        plt.figure('result' + self.name)
        plt.imshow(img)

        path = os.path.join(self.folder, '8bit', self.name + '.png')
        cv2.imwrite(path, self.gray_8_med_gau)
        path = os.path.join(self.folder, 'equalized', self.name + '.png')
        cv2.imwrite(path, self.equ_img)

    def ROI(self, all_step=True, debug=True):
        auto_mask, contours = self.msr(self.gray_8_med_gau)
        if (not os.path.exists(self.body_json_path)) or all_step:
            mask_pts = self.ROI_json(contours)
            self.mask_to_json(mask_pts, self.gray_8, self.ROI_json_path)
            os.popen('labelme ' + self.ROI_json_path).read()
        mask_data = self.labelme_ROI_mask(self.ROI_json_path, self.gray_16_gau)
        if mask_data:
            self.mask_data = self.process_ROI(mask_data)
            result = self.stat_ROI(self.mask_data)
        else:
            result=pd.DataFrame()
            self.mask_data={'le':np.zeros_like(self.gray_8)}
            mask_data = {'le': np.zeros_like(self.gray_8)}

        if debug:
            color = {'le': [0], 'lc': [1], 'rc': [2], 're': [0, 1]}
            fig = plt.figure(self.name, figsize=(7, 7))
            merg_mask = self.get_merg_mask(self.color_8, auto_mask)
            plt.subplot(221)
            plt.imshow(merg_mask)
            plt.title('auto ROI')
            merg_mask = self.color_8
            for key, value in mask_data.items():
                merg_mask = self.get_merg_mask(merg_mask, value, index=color[key])
            plt.subplot(222)
            plt.imshow(merg_mask)
            plt.title('after lableme')
            merg_mask2 = self.color_8
            for key, value in self.mask_data.items():
                merg_mask2 = self.get_merg_mask(merg_mask2, value, index=color[key])
            plt.subplot(223)
            plt.imshow(merg_mask2)
            plt.title('final ROI')

            merg_mask3 = np.zeros_like(self.color_8)
            for key, value in self.mask_data.items():
                labelme = mask_data[key]
                merg_mask3 = self.get_merg_mask(merg_mask3, labelme - value, index=color[key])
            plt.subplot(224)
            plt.imshow(merg_mask3)
            plt.title('diff')
            plt.savefig(os.path.join(self.folder, 'debug', self.name + '_debug.png'))


            # plt.close(fig)

        return result

        if debug:
            merg_mask = self.get_merg_mask(self.color_8, auto_mask)
            self.show_img(merg_mask, name='auto ROI')

    def body(self, all_step=True, debug=False):

        mask, contour = self.get_body_mask(self.gray_8_med_gau)
        if (not os.path.exists(self.body_json_path)) or all_step:
            self.body_json(contour)
            # subprocess.Popen('labelme '+self.body_json_path)
            os.popen('labelme ' + self.body_json_path).read()
        mask_body, masked_body = self.labelme_body_mask(self.body_json_path, self.gray_16_gau)


        if debug:
            plt.figure(figsize=(7, 7))
            merg_mask = self.get_merg_mask(self.equ_img_color, mask)
            plt.subplot(221)
            plt.imshow(merg_mask)
            plt.title('auto body')
            plt.subplot(222)
            plt.imshow(mask)
            plt.title('auto mask')

            plt.subplot(223)
            plt.imshow(masked_body)
            plt.title('masked 16 bit img')
            plt.subplot(224)
            plt.imshow(mask_body)
            plt.title('mask after labelme')
        return mask_body, masked_body

    def read_img(self):
        self.gray_16 = cv2.imread(self.path, -1)
        self.color_8 = cv2.imread(self.path)
        self.gray_8 = cv2.cvtColor(self.color_8, cv2.COLOR_BGR2GRAY)
        self.gray_8_med = cv2.medianBlur(self.gray_8, 11)
        self.gray_8_med_gau = cv2.GaussianBlur(self.gray_8_med, (5, 5), 0)
        # self.gray_16_med= cv2.medianBlur(self.gray_16, 11)
        self.gray_16_gau = cv2.GaussianBlur(self.gray_16, (5, 5), 0)
        self.equ_img = cv2.equalizeHist(self.gray_8_med_gau)
        self.equ_img_color = cv2.merge([self.equ_img, self.equ_img, self.equ_img])
        # with open(self.path,'rb') as image_file:
        # image_text_data = image_file.read()

    def msr(self, gray, min_area=80, max_area=300):
        mser = cv2.MSER_create(min_area=min_area, max_area=max_area)
        regions, boxes = mser.detectRegions(gray)

        mask = np.zeros(np.shape(gray), dtype=np.uint8)
        for px in regions:
            mask[px[:, 1], px[:, 0]] = 225
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        '''
        if len(boxes) > 0:
            start_x = boxes[:, 0]
            start_y = boxes[:, 1]
            w_x = boxes[:, 2]
            h_y = boxes[:, 3]

            for x, y, w, h in zip(start_x, start_y, w_x, h_y):
                mask[y:y + h - 1, x:x + w - 1, ] = 255

            mser = cv2.MSER_create(min_area=min_area, max_area=max_area)
            regions, boxes = mser.detectRegions(mask)
            '''

        return mask, contours

    def show_img(self, src, name='image'):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, src.shape[1], src.shape[0]);
        cv2.imshow(name, src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_boxes(self, src, boxes, name='boxes'):
        img = src.copy()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, img.shape[1], img.shape[0]);
        for box in boxes:
            x0, y0, w0, h0 = box
            cv2.rectangle(img, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 1)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_contours(self, src, contours, name='contours'):
        img = src.copy()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, src.shape[1], src.shape[0]);
        for contours in contours:
            cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_merg_mask(self, cimg, mask, index=[1]):
        zero_mask = np.zeros((mask.shape), dtype=np.uint8)
        cbd_mask = cv2.merge([zero_mask, zero_mask, zero_mask])
        for k in index:
            cbd_mask[:, :, k] = mask
        cimg = cv2.addWeighted(cimg, 1, cbd_mask, 0.3, 0)
        return cimg

    def get_body_mask(self, gray):
        lbg = np.mean(gray[0:50, 0:50])
        body1 = gray > 15 * lbg
        body_mask = np.zeros((body1.shape), dtype=np.uint8)
        body_mask[body1] = 255
        contours, hierarchy = cv2.findContours(body_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for k in range(len(contours)):
            areas.append(cv2.contourArea(contours[k]))
            max_idx = np.argmax(np.array(areas))
        contour = contours[max_idx]
        area = areas[max_idx]
        body_mask = np.zeros_like(body_mask)

        contour = np.reshape(contour, (-1, 2))
        cv2.fillPoly(body_mask, [contour.astype(np.int)], 255)
        pts = contour[0]
        for pt in contour:
            dis = np.sqrt(np.square(pt - pts[-1]).sum())
            if dis > 10:
                pts = np.vstack((pts, pt))
        return body_mask, pts

    def body_json(self, contour):

        global labels
        pts = contour.reshape(-1, 2).tolist()
        mask_pt = {'label': labels[1],
                   'points': pts,
                   'group_id': None,
                   'shape_type': 'polygon',
                   'flags': {'auto': 1}}
        mask_pts = [mask_pt]
        self.mask_to_json(mask_pts, self.equ_img, self.body_json_path)

    def ROI_json(self, contours):
        rc = 70
        bodyc = [300, 300]

        ROI_dict = []
        centers = []
        global labels
        if contours:
            for contour in contours:
                contour = np.reshape(contour, (-1, 2))
                center = np.mean(contour, axis=0)
                centers.append(center)
                dis = center[0] - bodyc[0]

                if dis < -rc:  # left
                    label = labels[2]
                elif dis < 0:
                    label = labels[3]
                elif dis < rc:
                    label = labels[4]
                else:
                    label = labels[5]
                pts = contour.tolist()

                mask_pt = {'label': label,
                       'points': pts,
                       'group_id': None,
                       'shape_type': 'polygon',
                       'flags': {'auto': 1, 'center_x': center[0], 'center_y': center[1]}}
            ROI_dict.append(mask_pt)
        return ROI_dict

    def mask_to_json(self, mask_pts, mat, json_data_path):
        img_name = self.name
        json_folder = self.folder
        img_type = self.type

        with open(json_data_path, "w") as jsonFile:
            json_data = {'version': '4.5.9',
                         'flags': {}, 'shapes': mask_pts,
                         'imagePath': self.path,
                         'imageData': base64.b64encode(cv2.imencode('.png', mat)[1].tobytes()).decode('utf-8'),
                         'imageHeight': mat.shape[0],
                         'imageWidth': mat.shape[1]}
            json.dump(json_data, jsonFile, ensure_ascii=False)  # 将修改好的json文件写入本地，json.dump()函数，用于json文件的output

    def labelme_body_mask(self, json_path, img):
        with open(json_path, "r", encoding='utf-8') as jsonFile:  # 读取json文件
            json_data = json.load(jsonFile)  # json.load（）用于读取json文件函数
        mask_name = json_data['shapes'][0]['label']
        mask_pts = json_data['shapes'][0]['points']
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [np.array(mask_pts).astype(np.int)], 255)
        mask_bool = mask > 0
        mask_img = mask_bool * img
        return mask_bool, mask_img

    def labelme_ROI_mask(self, json_path, img):
        with open(json_path, "r", encoding='utf-8') as jsonFile:  # 读取json文件
            json_data = json.load(jsonFile)  # json.load（）用于读取json文件函数
        ROIs = json_data['shapes']
        mask_data = {}
        if ROIs:
            for ROI in ROIs:
                mask_name = ROI['label']
                mask_pts = ROI['points']
                mask = np.zeros_like(img)
                cv2.fillPoly(mask, [np.array(mask_pts).astype(np.int)], 255)
                mask_bool = mask > 0
                mask_img = mask_bool * img
                mask_data[mask_name] = mask_img
                # print(mask_name)
        return mask_data

    def process_ROI(self, mask_data, debug=False):
        k=-50
        th=0.5
        for key, value in mask_data.items():
            name = os.path.join(self.folder, 'debug', self.name+'_' +key+ '.png')
            cv2.imwrite(name, value)

            sort_value = value.copy().flatten()
            sort_value = np.sort(sort_value)
            ref = sort_value[k]
            thresh = th * ref
            masked = value > thresh
            data = masked * value
            mask_data[key] = data
            if debug:
                print(thresh)
                plt.figure('process ROI')
                plt.subplot(221)
                plt.imshow(value)
                plt.title('ROI read labelme')
                plt.subplot(222)
                plt.imshow(data)
                plt.title('ROI after process')
                plt.subplot(223)
                plt.imshow(data - value)
                plt.title('diff')
        return mask_data

    def stat_ROI(self, mask_data):
        count = 0
        result = pd.DataFrame(columns=column_names)
        for key, value in mask_data.items():
            result.loc[count, column_names[0]] = self.name
            result.loc[count, column_names[1]] = self.good_not(self.name, key)
            result.loc[count, column_names[2]] = key
            y, x = np.where(value > 0)
            result.loc[count, column_names[3]] = int(np.mean(x))
            result.loc[count, column_names[4]] = int(np.mean(y))
            result.loc[count, column_names[5]] = np.max(value)
            #mask = value > 0
            mask = np.zeros_like(value)
            mask[value > 0] = 1
            result.loc[count, column_names[6]] = mask.sum()
            result.loc[count, column_names[7]] = value
            count = count + 1
        return result

    def good_not(self, name, label):
        if (not (name[0] == '0')) and (label[0] == 'l'):
            flag = 'big'
        else:
            flag = 'small'
        return flag

    def predict(self, results):
        try:
            #f = open('thresh.txt', mode='r')
            f = open(os.path.join(self.folder, 'result.txt'), mode='r')
            thresh = np.double(f.read())
            f.close()

            for index, row in results.iterrows():
                if row['area'] > thresh:
                    pre = 'abnormal'
                else:
                    pre = 'normal'
                results.loc[index, 'predict'] = pre
                print("%s %s area is: %d, predic: %s, thresh: %d" % (
                row['image'], row['lympha'], row['area'], row['predict'], thresh))
            return results

        except:
            print('no thresh.txt! please form thresh first')


def plot_result(results, folder):
    small_group = results[results['big_small'] == 'small']
    big_group = results[results['big_small'] == 'big']

    small_group_data = small_group['area']
    big_group_data = big_group['area']

    fig = plt.figure('final hist')
    # areas=[pd.concat([small_group_data,big_group_data]).tolist()]
    areas = np.array(pd.concat([small_group_data, big_group_data]))
    plt.hist(areas, bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel('Lympha area')
    plt.ylabel('count')
    plt.savefig(os.path.join(folder, 'result', 'hist.png'))
    #thresh = filters.threshold_otsu(areas)
    #thresh, otsu = cv2.threshold(areas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if len(areas)>1:
        kmeans = KMeans(n_clusters=2).fit(areas.reshape(-1, 1))
        thresh=np.mean(kmeans.cluster_centers_)
    else:
        print('error less that 2 sample!')
        thresh=areas
    plt.close(fig)

    fig = plt.figure('final result')
    small_group_data = small_group['area']
    big_group_data = big_group['area']
    plt.scatter(np.random.rand(len(small_group_data)) + 0.5, small_group_data, c='k')
    plt.scatter(np.random.rand(len(big_group_data)) + 2.5, big_group_data, c='b')
    plt.plot([0.5, 3.5], [thresh, thresh], 'r--')
    plt.errorbar(x=1, y=np.mean(small_group_data), yerr=np.std(small_group_data), fmt='k')
    plt.errorbar(x=3, y=np.mean(big_group_data), yerr=np.std(big_group_data), fmt='b')
    plt.xticks([1, 3], ['normal', 'abnormal'])
    plt.xlabel('class')
    plt.ylabel('Lympha area')
    plt.savefig(os.path.join(folder, 'result', 'scater.png'))
    plt.close(fig)

    return thresh


def make_all_dir(labels, data_path):
    try:
        os.makedirs(os.path.join(data_path, 'equalized', 'labelme'))
    except:
        pass
    try:
        os.makedirs(os.path.join(data_path, 'equalized'))
    except:
        pass
    try:
        os.makedirs(os.path.join(data_path, '8bit'))
    except:
        pass
    try:
        os.makedirs(os.path.join(data_path, '8bit', 'labelme'))
    except:
        pass
    try:
        os.makedirs(os.path.join(data_path, 'mask', 'merge_color_total'))
    except:
        pass
    try:
        os.makedirs(os.path.join(data_path, 'mask', 'total'))
    except:
        pass
    try:
        os.makedirs(os.path.join(data_path, 'mask', 'lympha'))
    except:
        pass
    try:
        os.makedirs(os.path.join(data_path, 'debug'))
    except:
        pass
    try:
        os.makedirs(os.path.join(data_path, 'result'))
    except:
        pass

    for label in labels:
        try:
            os.makedirs(os.path.join(data_path, 'mask', label))
        except:
            pass

if __name__ == '__main__':

    data_path = '.\data\\test'


    img_type = "*.tif"
    column_names = ['image', 'big_small', 'lympha', 'x_center', 'y_center', 'max_intenisty', 'area', 'ROI']
    labels = ['bg', 'mouse', 'le', 'lc', 'rc', 're']
    make_all_dir(labels, data_path)

    imgs_path = glob.glob(os.path.join(data_path, img_type))
    results = pd.DataFrame(columns=column_names)
    str = input("if predict, please press enter. get thresh please press other keys then press enter")
    if str == '':

        print("----- go to predict images----- ")
        predict = True
        excel_name = 'result_pred.xls'
    else:
        print('****go to statistic images  *****')
        predict = False
        excel_name = 'result.xls'

    for img_path in imgs_path:
        img_name = img_path.split('\\')[-1].split('.')[0]

        print(img_name)
        mouse_img = mouse(img_path)
        mouse_img.pipline()
        if predict:
            mouse_img.result = mouse_img.predict(mouse_img.result)
        results = pd.concat([results, mouse_img.result], axis=0)

        results.to_excel(os.path.join(data_path,'result', excel_name))

    if not predict:
        thresh = plot_result(results, data_path)
        f = open(os.path.join(data_path,'result','thresh.txt'), mode='w')
        f.write('%f'%thresh)
        f.close()


