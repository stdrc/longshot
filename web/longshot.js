/**
 * 长截图拼接工具 - JavaScript 版本
 * 使用最长公共子串算法找到图片重叠部分并进行拼接
 */

class LongScreenshotStitcher {
    constructor() {
        this.ignoreRightPixels = 20; // 忽略右侧像素数量，用于排除滚动条影响
    }

    /**
     * 将图片的每一行转换为哈希值，用于快速比较
     * @param {ImageData} imageData - Canvas ImageData 对象
     * @param {number} ignoreRightPixels - 忽略右侧多少像素
     * @returns {Array<number>} 行哈希数组
     */
    imageToRowHashes(imageData, ignoreRightPixels = 20) {
        const { width, height, data } = imageData;
        const rowHashes = [];
        const effectiveWidth = Math.max(0, width - ignoreRightPixels);

        // 批量处理以提高性能
        for (let y = 0; y < height; y++) {
            const rowStart = y * width * 4;

            // 使用更高效的采样策略 - 每隔几个像素采样一次
            const sampleStep = Math.max(1, Math.floor(effectiveWidth / 100)); // 最多采样100个点
            let sumR = 0, sumG = 0, sumB = 0;
            let sampleCount = 0;

            for (let x = 0; x < effectiveWidth; x += sampleStep) {
                const pixelIndex = rowStart + x * 4;
                sumR += data[pixelIndex];     // R
                sumG += data[pixelIndex + 1]; // G
                sumB += data[pixelIndex + 2]; // B
                sampleCount++;
            }

            if (sampleCount > 0) {
                // 计算平均值并量化以提高容忍度
                const avgR = Math.floor((sumR / sampleCount) / 16) * 16; // 增大量化步长
                const avgG = Math.floor((sumG / sampleCount) / 16) * 16;
                const avgB = Math.floor((sumB / sampleCount) / 16) * 16;

                // 更高效的哈希函数
                const hash = (avgR << 16) | (avgG << 8) | avgB;
                rowHashes.push(hash);
            } else {
                rowHashes.push(0);
            }
        }

        return rowHashes;
    }

    /**
     * 找到两个序列的最长公共子串
     * @param {Array<number>} seq1 - 第一个序列
     * @param {Array<number>} seq2 - 第二个序列
     * @param {number} minRatio - 最小重叠比例阈值
     * @returns {Object} {seq1Start, seq2Start, length}
     */
    findLongestCommonSubstring(seq1, seq2, minRatio = 0.01) {
        const m = seq1.length;
        const n = seq2.length;
        const minLength = Math.floor(Math.min(m, n) * minRatio);

        // 动态规划表
        const dp = Array(m + 1).fill(0).map(() => Array(n + 1).fill(0));

        let maxLength = 0;
        let endingPosI = 0;
        let endingPosJ = 0;

        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (seq1[i - 1] === seq2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    if (dp[i][j] > maxLength) {
                        maxLength = dp[i][j];
                        endingPosI = i;
                        endingPosJ = j;
                    }
                } else {
                    dp[i][j] = 0;
                }
            }
        }

        if (maxLength < minLength) {
            return { seq1Start: -1, seq2Start: -1, length: 0 };
        }

        const seq1Start = endingPosI - maxLength;
        const seq2Start = endingPosJ - maxLength;

        return { seq1Start, seq2Start, length: maxLength };
    }

    /**
     * 寻找最佳重叠区域
     * @param {Array<number>} img1Hashes - 第一张图片的行哈希
     * @param {Array<number>} img2Hashes - 第二张图片的行哈希
     * @returns {Object} 重叠信息
     */
    findBestOverlap(img1Hashes, img2Hashes) {
        console.log(`搜索重叠区域: img1有${img1Hashes.length}行, img2有${img2Hashes.length}行`);

        const overlap = this.findLongestCommonSubstring(img1Hashes, img2Hashes, 0.01);

        if (overlap.length > 0) {
            const overlapRatio = overlap.length / Math.min(img1Hashes.length, img2Hashes.length);
            console.log(`找到重叠: 长度${overlap.length}行, 占比${(overlapRatio * 100).toFixed(2)}%`);
            return overlap;
        } else {
            console.log("未找到任何重叠区域");
            return { seq1Start: -1, seq2Start: -1, length: 0 };
        }
    }

    /**
     * 将图片绘制到 Canvas 并获取 ImageData
     * @param {HTMLImageElement} img - 图片元素
     * @returns {Object} {canvas, ctx, imageData}
     */
    imageToImageData(img) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', {
            alpha: false,  // 禁用 alpha 通道以提高性能
            desynchronized: true  // 启用异步渲染
        });

        canvas.width = img.width;
        canvas.height = img.height;

        // 使用高质量图像平滑算法
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        return { canvas, ctx, imageData };
    }

    /**
     * 拼接两张图片
     * @param {HTMLImageElement} img1 - 第一张图片
     * @param {HTMLImageElement} img2 - 第二张图片
     * @returns {Promise<HTMLCanvasElement>} 拼接结果的 Canvas
     */
    async stitchImages(img1, img2) {
        console.log(`处理图片: ${img1.width}x${img1.height} + ${img2.width}x${img2.height}`);

        // 确保两张图片宽度相同
        let processedImg1 = img1;
        if (img1.width !== img2.width) {
            console.log(`调整图片宽度: ${img1.width} -> ${img2.width}`);
            processedImg1 = await this.resizeImage(img1, img2.width, Math.floor(img1.height * img2.width / img1.width));
        }

        // 获取图片数据
        const img1Data = this.imageToImageData(processedImg1);
        const img2Data = this.imageToImageData(img2);

        console.log(`忽略右侧 ${this.ignoreRightPixels} 像素来排除滚动条影响`);

        // 转换为行哈希
        const img1Hashes = this.imageToRowHashes(img1Data.imageData, this.ignoreRightPixels);
        const img2Hashes = this.imageToRowHashes(img2Data.imageData, this.ignoreRightPixels);

        // 寻找重叠区域
        const overlap = this.findBestOverlap(img1Hashes, img2Hashes);

        // 创建结果 Canvas，优化设置
        const resultCanvas = document.createElement('canvas');
        const resultCtx = resultCanvas.getContext('2d', {
            alpha: false,
            desynchronized: true
        });

        if (overlap.length === 0) {
            console.log("未找到重叠区域，直接拼接");
            // 直接拼接
            resultCanvas.width = img2.width;
            resultCanvas.height = processedImg1.height + img2.height;

            resultCtx.drawImage(processedImg1, 0, 0);
            resultCtx.drawImage(img2, 0, processedImg1.height);
        } else {
            const { seq1Start: img1Start, seq2Start: img2Start, length: overlapLength } = overlap;
            console.log(`找到重叠区域: img1[${img1Start}:${img1Start + overlapLength}] = img2[${img2Start}:${img2Start + overlapLength}]`);

            // 计算拼接后的总高度
            const img1KeepHeight = img1Start + overlapLength;
            const img2SkipHeight = img2Start + overlapLength;
            const img2KeepHeight = img2.height - img2SkipHeight;
            const resultHeight = img1KeepHeight + img2KeepHeight;

            console.log(`拼接计算: img1保留${img1KeepHeight}行 + img2跳过${img2SkipHeight}行保留${img2KeepHeight}行 = 总计${resultHeight}行`);

            resultCanvas.width = img2.width;
            resultCanvas.height = resultHeight;

            // 绘制img1的保留部分
            resultCtx.drawImage(
                processedImg1,
                0, 0, processedImg1.width, img1KeepHeight,
                0, 0, processedImg1.width, img1KeepHeight
            );

            // 绘制img2的剩余部分
            if (img2KeepHeight > 0) {
                resultCtx.drawImage(
                    img2,
                    0, img2SkipHeight, img2.width, img2KeepHeight,
                    0, img1KeepHeight, img2.width, img2KeepHeight
                );
            }
        }

        return resultCanvas;
    }

    /**
     * 调整图片尺寸
     * @param {HTMLImageElement} img - 原图片
     * @param {number} newWidth - 新宽度
     * @param {number} newHeight - 新高度
     * @returns {Promise<HTMLImageElement>} 调整后的图片
     */
    async resizeImage(img, newWidth, newHeight) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = newWidth;
            canvas.height = newHeight;

            ctx.drawImage(img, 0, 0, newWidth, newHeight);

            const resizedImg = new Image();
            resizedImg.onload = () => resolve(resizedImg);
            resizedImg.src = canvas.toDataURL();
        });
    }

    /**
     * 拼接多张图片
     * @param {Array<File>} imageFiles - 图片文件数组
     * @param {Function} progressCallback - 进度回调函数
     * @returns {Promise<HTMLCanvasElement>} 拼接结果的 Canvas
     */
    async stitchMultipleImages(imageFiles, progressCallback = null) {
        if (imageFiles.length < 2) {
            throw new Error("至少需要两张图片进行拼接");
        }

        console.log(`开始拼接 ${imageFiles.length} 张图片...`);

        // 流式处理图片，避免同时加载所有图片占用过多内存
        let result = await this.loadImage(imageFiles[0]);
        console.log(`基础图片: ${result.width}x${result.height}`);

        if (progressCallback) {
            progressCallback({
                stage: 'loading',
                current: 1,
                total: imageFiles.length
            });
        }

        for (let i = 1; i < imageFiles.length; i++) {
            console.log(`\n处理第 ${i + 1} 张图片`);

            // 逐个加载图片，处理完立即释放
            const nextImg = await this.loadImage(imageFiles[i]);

            if (progressCallback) {
                progressCallback({
                    stage: 'loading',
                    current: i + 1,
                    total: imageFiles.length
                });
            }

            // 如果 result 是 Canvas，需要转换为 Image
            if (result instanceof HTMLCanvasElement) {
                const tempImg = await this.canvasToImage(result);
                result = tempImg;
            }

            // 执行拼接
            const newResult = await this.stitchImages(result, nextImg);

            // 释放之前的结果以节省内存
            if (result instanceof HTMLImageElement && result.src.startsWith('blob:')) {
                URL.revokeObjectURL(result.src);
            }

            result = newResult;
            console.log(`当前结果尺寸: ${result.width}x${result.height}`);

            if (progressCallback) {
                progressCallback({
                    stage: 'stitching',
                    current: i,
                    total: imageFiles.length - 1
                });
            }

            // 强制垃圾回收 (如果浏览器支持)
            if (window.gc) {
                window.gc();
            }
        }

        console.log(`拼接完成! 最终尺寸: ${result.width}x${result.height}`);
        return result;
    }

    /**
     * 将 Canvas 转换为 Image
     * @param {HTMLCanvasElement} canvas - Canvas 元素
     * @returns {Promise<HTMLImageElement>} 转换后的图片
     */
    async canvasToImage(canvas) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.src = canvas.toDataURL('image/jpeg', 0.9); // 使用 JPEG 压缩以节省内存
        });
    }

    /**
     * 从文件加载图片
     * @param {File} file - 图片文件
     * @returns {Promise<HTMLImageElement>} 加载的图片
     */
    loadImage(file) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            const url = URL.createObjectURL(file);

            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
            };

            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error(`无法加载图片: ${file.name}`));
            };

            img.src = url;
        });
    }

    /**
     * 设置忽略右侧像素数量
     * @param {number} pixels - 像素数量
     */
    setIgnoreRightPixels(pixels) {
        this.ignoreRightPixels = Math.max(0, pixels);
    }
}

// 导出类供其他脚本使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LongScreenshotStitcher;
}
