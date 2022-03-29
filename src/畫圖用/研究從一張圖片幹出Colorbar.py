import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 讀入一張要判斷色階的圖片 (colorbar)
    raw = cv2.imread("Adobe_colorbar_non_Sort.png", cv2.IMREAD_COLOR)
    res = np.zeros(shape=raw.shape, dtype=raw.dtype)

    # 拆 BGR 通道
    B, G, R = raw[::, ::, 0], raw[::, ::, 1], raw[::, ::, 2]
    # 計算亮度是用浮點數
    B, G, R = B / 255., G / 255., R / 255.
    # 公式參照 : HSP Color Model — Alternative to HSV (HSB) and HSL
    # url: https://alienryderflex.com/hsp.html
    L = np.sqrt(0.299 * R ** 2 + 0.587 * G ** 2 + 0.114 * B ** 2)  # 計算感知亮度公式

    _flt_L_idx = np.argsort(L.reshape(-1))  # 算好排序過的 IDX
    _flt_raw = raw.reshape((1, -1, 3))  # 統一展平
    _flt_res = res.reshape((1, -1, 3))  # 統一展平

    fig = plt.figure(figsize=(8, 6))  # 屌畫

    # 尚未排序的
    ax_non = fig.add_subplot(221)
    ax_non.set_title("1. Non-sorted")
    ax_non.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))


    # 排序好的 IDX 跟 走訪用的不要搞混。。。
    for i, sort_idx in enumerate(_flt_L_idx):
        _flt_res[0][i] = _flt_raw[0, sort_idx]

    _flt_result = _flt_res.copy()  # 後續計算不重複色階用
    # 展示排序結果用，注意是1D排好，畫2D顯示，所以亮度會呈現斜斜的
    _flt_res = _flt_res.reshape(raw.shape)

    # 呈現排好的 2D
    ax_sorted = fig.add_subplot(223)
    ax_sorted.set_title("2. Sorted")
    ax_sorted.imshow(cv2.cvtColor(_flt_res, cv2.COLOR_BGR2RGB))

    # 儲存
    cv2.imwrite("Adobe_colorbar_2d_Sorted.png", _flt_res)

    # 接下來計算 不重複元素構成的 colorbar
    all_diff_result = []

    # _flt_result.shape = (1, 5202, 3)
    for i in range(_flt_result.shape[1]):
        if len(all_diff_result) == 0:
            all_diff_result.append(_flt_result[0, i])

        else:
            if np.all(all_diff_result[-1] == _flt_result[0, i]):  # 上一個元素跟這次要加上的元素相同的話
                pass
            else:
                all_diff_result.append(_flt_result[0, i])
    all_diff_result = np.array(all_diff_result, np.uint8)

    # 已經計算出不重複的，但是資料很細。
    all_diff_result = all_diff_result.reshape(-1, 1, 3)
    all_diff_result.shape

    #  加粗，注意是 指數次方增粗 (粗度倍率 = 2^thickness)
    thickness = 4
    for i in range(thickness):
        all_diff_result = np.hstack((all_diff_result, all_diff_result))

    # 存檔
    cv2.imwrite("Adobe_colorbar_nonRepeat.png", all_diff_result)

    colorbar = fig.add_subplot(122)
    colorbar.axison = False
    colorbar.set_title("3. non-repeat colorbar")
    colorbar.imshow(cv2.cvtColor(all_diff_result, cv2.COLOR_BGR2RGB))
    plt.savefig("提取Colorbar流程圖.png")
