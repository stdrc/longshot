#!/usr/bin/env python3
"""
长截图拼接脚本
使用最长公共子串算法找到图片重叠部分并进行拼接
"""

import numpy as np
from PIL import Image
import os
from typing import List, Tuple, Optional


def image_to_row_hashes(image: Image.Image, ignore_right_pixels: int = 20) -> List[int]:
    """
    将图片的每一行转换为哈希值，用于快速比较
    ignore_right_pixels: 忽略右侧多少像素（用于排除滚动条影响）
    """
    img_array = np.array(image)
    row_hashes = []

    for row in img_array:
        # 去掉右侧的像素以忽略滚动条
        if ignore_right_pixels > 0 and row.shape[0] > ignore_right_pixels:
            row_trimmed = row[:-ignore_right_pixels]
        else:
            row_trimmed = row

        # 使用更加鲁棒的哈希算法 - 计算行的平均色彩值
        # 这样可以容忍小的像素差异（比如压缩造成的微小变化）
        if len(row_trimmed.shape) == 2:  # RGB图像
            row_mean = np.mean(row_trimmed, axis=0)
            # 量化到较少的级别以提高容忍度
            row_quantized = (row_mean / 8).astype(int) * 8
            row_hash = hash(row_quantized.tobytes())
        else:
            # 灰度图像
            row_mean = np.mean(row_trimmed)
            row_quantized = int(row_mean / 8) * 8
            row_hash = hash(str(row_quantized))

        row_hashes.append(row_hash)

    return row_hashes


def find_longest_common_substring(
    seq1: List[int], seq2: List[int], min_ratio: float = 0.1
) -> Tuple[int, int, int]:
    """
    找到两个序列的最长公共子串
    返回 (seq1_start, seq2_start, length)
    min_ratio: 最小重叠比例阈值（相对于较短图片的高度）
    """
    m, n = len(seq1), len(seq2)
    min_length = int(min(m, n) * min_ratio)

    # 动态规划表
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    max_length = 0
    ending_pos_i = 0
    ending_pos_j = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    ending_pos_i = i
                    ending_pos_j = j
            else:
                dp[i][j] = 0

    if max_length < min_length:
        return -1, -1, 0

    # 计算起始位置
    start_i = ending_pos_i - max_length
    start_j = ending_pos_j - max_length

    return start_i, start_j, max_length


def find_best_overlap(
    img1_hashes: List[int], img2_hashes: List[int]
) -> Tuple[int, int, int]:
    """
    寻找最佳重叠区域
    直接在整张图片上寻找最长公共子串
    """
    print(f"  搜索重叠区域: img1有{len(img1_hashes)}行, img2有{len(img2_hashes)}行")

    # 先尝试更低的阈值
    overlap = find_longest_common_substring(img1_hashes, img2_hashes, min_ratio=0.01)

    if overlap[2] > 0:
        overlap_ratio = overlap[2] / min(len(img1_hashes), len(img2_hashes))
        print(f"  找到重叠: 长度{overlap[2]}行, 占比{overlap_ratio:.2%}")
        return overlap
    else:
        print("  未找到任何重叠区域")
        return (-1, -1, 0)


def stitch_images(
    img1: Image.Image, img2: Image.Image, ignore_right_pixels: int = 20
) -> Optional[Image.Image]:
    """
    拼接两张图片
    ignore_right_pixels: 忽略右侧多少像素（用于排除滚动条影响）
    """
    print(f"处理图片: {img1.size} + {img2.size}")

    # 确保两张图片宽度相同
    if img1.width != img2.width:
        print(f"调整图片宽度: {img1.width} -> {img2.width}")
        img1 = img1.resize(
            (img2.width, int(img1.height * img2.width / img1.width)),
            Image.Resampling.LANCZOS,
        )

    # 转换为行哈希（忽略右侧像素以排除滚动条影响）
    print(f"忽略右侧 {ignore_right_pixels} 像素来排除滚动条影响")
    img1_hashes = image_to_row_hashes(img1, ignore_right_pixels)
    img2_hashes = image_to_row_hashes(img2, ignore_right_pixels)

    # 寻找重叠区域
    overlap = find_best_overlap(img1_hashes, img2_hashes)

    if overlap[2] == 0:
        print("未找到重叠区域，直接拼接")
        # 如果没有重叠，直接拼接
        result_height = img1.height + img2.height
        result = Image.new("RGB", (img1.width, result_height))
        result.paste(img1, (0, 0))
        result.paste(img2, (0, img1.height))
        return result

    img1_start, img2_start, overlap_length = overlap
    print(
        f"找到重叠区域: img1[{img1_start}:{img1_start + overlap_length}] = img2[{img2_start}:{img2_start + overlap_length}]"
    )

    # 计算拼接后的总高度
    img1_keep_height = img1_start + overlap_length  # 保留img1的部分
    img2_skip_height = img2_start + overlap_length  # 跳过img2的重叠部分
    img2_keep_height = img2.height - img2_skip_height  # 保留img2的剩余部分

    result_height = img1_keep_height + img2_keep_height

    print(
        f"拼接计算: img1保留{img1_keep_height}行 + img2跳过{img2_skip_height}行保留{img2_keep_height}行 = 总计{result_height}行"
    )

    # 创建结果图片
    result = Image.new("RGB", (img1.width, result_height))

    # 粘贴img1的保留部分
    img1_crop = img1.crop((0, 0, img1.width, img1_keep_height))
    result.paste(img1_crop, (0, 0))

    # 粘贴img2的剩余部分
    if img2_keep_height > 0:
        img2_crop = img2.crop((0, img2_skip_height, img2.width, img2.height))
        result.paste(img2_crop, (0, img1_keep_height))

    return result


def stitch_multiple_images(
    image_paths: List[str], output_path: str, ignore_right_pixels: int = 20
) -> None:
    """
    拼接多张图片
    ignore_right_pixels: 忽略右侧多少像素（用于排除滚动条影响）
    """
    if len(image_paths) < 2:
        print("至少需要两张图片进行拼接")
        return

    print(f"开始拼接 {len(image_paths)} 张图片...")

    # 加载第一张图片
    result = Image.open(image_paths[0])
    print(f"基础图片: {image_paths[0]} ({result.size})")

    # 逐个拼接后续图片
    for i, path in enumerate(image_paths[1:], 1):
        print(f"\n拼接第 {i+1} 张图片: {path}")
        next_img = Image.open(path)
        result = stitch_images(result, next_img, ignore_right_pixels)
        if result is None:
            print("拼接失败")
            return
        print(f"当前结果尺寸: {result.size}")

    # 保存结果
    result.save(output_path, "JPEG", quality=95)
    print(f"\n拼接完成! 结果已保存到: {output_path}")
    print(f"最终尺寸: {result.size}")


def main():
    """
    主函数
    """
    # 获取当前目录下的图片文件
    image_files = []
    # for file in ["1.jpeg", "2.jpeg"]:
    for file in [
        "IMG_8F9EF97CEBE1-1.jpeg",
        "IMG_8F9EF97CEBE1-2.jpeg",
        "IMG_8F9EF97CEBE1-3.jpeg",
        "IMG_8F9EF97CEBE1-4.jpeg",
        "IMG_8F9EF97CEBE1-5.jpeg",
    ]:
        if os.path.exists(file):
            image_files.append(file)
        else:
            print(f"文件不存在: {file}")

    if len(image_files) < 2:
        print("需要至少两张图片文件 (1.jpeg, 2.jpeg)")
        return

    # 按文件名排序
    image_files.sort()

    # 输出文件名
    output_file = "stitched_result.jpeg"

    # 执行拼接（忽略右侧20像素来排除滚动条影响）
    ignore_pixels = 20
    print(f"配置: 忽略右侧 {ignore_pixels} 像素以排除滚动条影响")
    stitch_multiple_images(image_files, output_file, ignore_pixels)


if __name__ == "__main__":
    main()
