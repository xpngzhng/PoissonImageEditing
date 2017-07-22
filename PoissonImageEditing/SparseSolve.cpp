#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct IndexedValue
{
    IndexedValue() : index(-1), value(0) {}
    IndexedValue(int index_, double value_) : index(index_), value(value_) {}
    int index;
    double value;
};

struct SparseMat
{
    SparseMat() : rows(0), maxCols(0) {}

    SparseMat(int rows_, int cols_) :
        rows(0), maxCols(0)
    {
        create(rows_, cols_);
    }

    void create(int rows_, int cols_)
    {
        CV_Assert(rows_ > 0 && cols_ > 0);

        rows = rows_;
        maxCols = cols_;
        buf.resize(rows * maxCols);
        data = &buf[0];
        memset(data, -1, rows * maxCols * sizeof(IndexedValue));
        count.resize(rows);
        memset(&count[0], 0, rows * sizeof(int));
    }

    void release()
    {
        rows = 0;
        maxCols = 0;
        buf.clear();
        count.clear();
        data = 0;
    }

    const IndexedValue* rowPtr(int row) const
    {
        CV_Assert(row >= 0 && row < rows);
        return data + row * maxCols;
    }

    IndexedValue* rowPtr(int row)
    {
        CV_Assert(row >= 0 && row < rows);
        return data + row * maxCols;
    }

    void insert(int row, int col, double value)
    {
        CV_Assert(row >= 0 && row < rows);

        int currCount = count[row];
        CV_Assert(currCount < maxCols);

        IndexedValue* rowData = rowPtr(row);        
        int i = 0;
        if ((currCount > 0) && (col > rowData[0].index))
        {
            for (i = 1; i < currCount; i++)
            {
                if ((col > rowData[i - 1].index) &&
                    (col < rowData[i].index))
                    break;
            }
        }
        if (i < currCount)
        {
            for (int j = currCount; j >= i; j--)
                rowData[j + 1] = rowData[j];
        }
        rowData[i] = IndexedValue(col, value);
        ++count[row];
    }

    void calcDiagonalElementsPositions(std::vector<int>& pos) const
    {
        pos.resize(rows, -1);
        for (int i = 0; i < rows; i++)
        {
            const IndexedValue* ptrRow = rowPtr(i);
            for (int j = 0; j < count[i]; j++)
            {
                if (ptrRow[j].index == i)
                {
                    pos[i] = j;
                    break;
                }
            }
        }
    }

    int rows, maxCols;    
    std::vector<IndexedValue> buf;
    std::vector<int> count;
    IndexedValue* data;

private:
    SparseMat(const SparseMat&);
    SparseMat& operator=(const SparseMat&);
};

void solve(const IndexedValue* A, const int* length, const int* diagPos, 
    const double* b, double* x, int rows, int cols, int maxIters, double eps)
{
    for (int iter = 0; iter < maxIters; iter++)
    {
        int count = 0;
        for (int i = 0; i < rows; i++)
        {
            double val = 0;
            const IndexedValue* ptrRow = A + cols * i;
            for (int j = 0; j < diagPos[i]; j++)
                val += ptrRow[j].value * x[ptrRow[j].index];
            for (int j = diagPos[i] + 1; j < length[i]; j++)
                val += ptrRow[j].value * x[ptrRow[j].index];
            val = b[i] - val;
            val /= ptrRow[diagPos[i]].value;
            if (fabs(val - x[i]) < eps)
                count++;
            x[i] = val;
        }
        if (count == rows)
        {
            printf("converge iter count = %d, end\n", iter + 1);
            break;
        }
    }
}

void makeIndex(const cv::Mat& mask, cv::Mat& index, int& numElems)
{
    CV_Assert(mask.data && mask.type() == CV_8UC1);
    
    int rows = mask.rows, cols = mask.cols;
    index.create(rows, cols, CV_32SC1);
    index.setTo(-1);
    int count = 0;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrMaskRow = mask.ptr<unsigned char>(i);
        int* ptrIndexRow = index.ptr<int>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMaskRow[j])
                ptrIndexRow[j] = (count++);
        }
    }
    numElems = count;
}

void draw(const std::vector<cv::Point>& contour, const cv::Size& imageSize, 
    cv::Rect& extendRect, cv::Mat& mask)
{
    cv::Rect contourRect = cv::boundingRect(contour);

    int left, right, top, bottom;
    left = contourRect.x;
    right = contourRect.x + contourRect.width;
    top = contourRect.y;
    bottom = contourRect.y + contourRect.height;
    CV_Assert(left > 0); 
    left--;
    CV_Assert(right < imageSize.width); 
    right++;
    CV_Assert(top > 0); 
    top--;
    CV_Assert(bottom < imageSize.height); 
    bottom++;

    extendRect.x = left;
    extendRect.y = top;
    extendRect.width = right - left;
    extendRect.height = bottom - top;

    mask.create(extendRect.height, extendRect.width, CV_8UC1);
    mask.setTo(0);
    std::vector<std::vector<cv::Point> > contours(1);
    contours[0] = contour;
    cv::drawContours(mask, contours, -1, cv::Scalar(255), -1, 8, 
        cv::noArray(), 0, cv::Point(-left, -top));
}

void draw(const std::vector<cv::Point>& contour, cv::Size& size, cv::Mat& mask)
{
    mask.create(size, CV_8UC1);
    mask.setTo(0);
    std::vector<std::vector<cv::Point> > contours(1);
    contours[0] = contour;
    cv::drawContours(mask, contours, -1, cv::Scalar(255), -1, 8, cv::noArray(), 0);
}

void getEquation(const cv::Mat& src, const cv::Mat& dst, 
    const cv::Mat& mask, const cv::Mat& index, int count,
    SparseMat& A, cv::Mat& b, cv::Mat& x, bool mixGrad = false)
{
    CV_Assert(src.data && dst.data && mask.data && index.data);
    CV_Assert((src.type() == CV_8UC1) && (dst.type() == CV_8UC1) &&
        (mask.type() == CV_8UC1) && (index.type() == CV_32SC1));
    CV_Assert((src.size() == dst.size()) && (src.size() == mask.size()) && 
        (src.size() == index.size()));
    
    int rows = src.rows, cols = src.cols;
    A.create(count, 8);
    b.create(count, 1, CV_64FC1);
    b.setTo(0);
    x.create(count, 1, CV_64FC1);
    x.setTo(0);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (mask.at<unsigned char>(i, j))
            {
                int currIndex = index.at<int>(i, j);
                int currSrcVal = src.at<unsigned char>(i, j);
                int currDstVal = dst.at<unsigned char>(i, j);
                int neighborCount = 0;
                int bVal = 0;
                if (i > 0)
                {
                    neighborCount++;
                    if (mask.at<unsigned char>(i - 1, j))
                    {
                        int topIndex = index.at<int>(i - 1, j);
                        A.insert(currIndex, topIndex, -1);
                    }
                    else
                    {
                        bVal += dst.at<unsigned char>(i - 1, j);
                    }
                    if (mixGrad)
                    {
                        int srcGrad = currSrcVal - src.at<unsigned char>(i - 1, j);
                        int dstGrad = currDstVal - dst.at<unsigned char>(i - 1, j);
                        bVal += (abs(srcGrad) > abs(dstGrad) ? srcGrad : dstGrad);
                    }
                    else
                        bVal += (currSrcVal - src.at<unsigned char>(i - 1, j));
                }
                if (i < rows - 1)
                {
                    neighborCount++;
                    if (mask.at<unsigned char>(i + 1, j))
                    {
                        int bottomIndex = index.at<int>(i + 1, j);
                        A.insert(currIndex, bottomIndex, -1);
                    }
                    else
                    {
                        bVal += dst.at<unsigned char>(i + 1, j);
                    }
                    if (mixGrad)
                    {
                        int srcGrad = currSrcVal - src.at<unsigned char>(i + 1, j);
                        int dstGrad = currDstVal - dst.at<unsigned char>(i + 1, j);
                        bVal += (abs(srcGrad) > abs(dstGrad) ? srcGrad : dstGrad);
                    }
                    else
                        bVal += (currSrcVal - src.at<unsigned char>(i + 1, j));
                }
                if (j > 0)
                {
                    neighborCount++;
                    if (mask.at<unsigned char>(i, j - 1))
                    {
                        int leftIndex = index.at<int>(i, j - 1);
                        A.insert(currIndex, leftIndex, -1);
                    }
                    else
                    {
                        bVal += dst.at<unsigned char>(i, j - 1);
                    }
                    if (mixGrad)
                    {
                        int srcGrad = currSrcVal - src.at<unsigned char>(i, j - 1);
                        int dstGrad = currDstVal - dst.at<unsigned char>(i, j - 1);
                        bVal += (abs(srcGrad) > abs(dstGrad) ? srcGrad : dstGrad);
                    }
                    else
                        bVal += (currSrcVal - src.at<unsigned char>(i, j - 1));
                }
                if (j < cols - 1)
                {
                    neighborCount++;
                    if (mask.at<unsigned char>(i, j + 1))
                    {
                        int rightIndex = index.at<int>(i, j + 1);
                        A.insert(currIndex, rightIndex, -1);
                    }
                    else
                    {
                        bVal += dst.at<unsigned char>(i, j + 1);
                    }
                    if (mixGrad)
                    {
                        int srcGrad = currSrcVal - src.at<unsigned char>(i, j + 1);
                        int dstGrad = currDstVal - dst.at<unsigned char>(i, j + 1);
                        bVal += (abs(srcGrad) > abs(dstGrad) ? srcGrad : dstGrad);
                    }
                    else
                        bVal += (currSrcVal - src.at<unsigned char>(i, j + 1));
                }
                A.insert(currIndex, currIndex, neighborCount);
                b.at<double>(currIndex) = bVal;
                x.at<double>(currIndex) = currSrcVal;
                //x.at<double>(currIndex) = dst.at<unsigned char>(i, j);
            }
        }
    }
}

void copy(const cv::Mat& val, const cv::Mat& mask, const cv::Mat& index, cv::Mat& dst)
{
    CV_Assert(val.data && val.type() == CV_64FC1);
    CV_Assert(mask.data && index.data && dst.data);
    CV_Assert((mask.type() == CV_8UC1) && (index.type() == CV_32SC1) && (dst.type() == CV_8UC1));
    CV_Assert((mask.size() == index.size()) && (mask.size() == dst.size()));

    int rows = mask.rows, cols = mask.cols;
    for (int i = 0; i < rows; i++)
    {
        const unsigned char* ptrMaskRow = mask.ptr<unsigned char>(i);
        const int* ptrIndexRow = index.ptr<int>(i);
        unsigned char* ptrDstRow = dst.ptr<unsigned char>(i);
        for (int j = 0; j < cols; j++)
        {
            if (ptrMaskRow[j])
            {
                ptrDstRow[j] = cv::saturate_cast<unsigned char>(val.at<double>(ptrIndexRow[j]));
            }
        }
    }
}

/*!
  The basic Poisson image editing function.
  Source image, mask image and destination image should have the same size.
  Source image's content inside the mask's non zero region will be blended into 
  the destination image, using Possion image editing algorithm.
  \param[in] src      Source image, should be of type CV_8UC1 or CV_8UC3.
  \param[in] mask     Source image's mask. Source image's content inside the mask's 
                      non zero region will be blended into the destination image.
                      The mask's non zero region should not include the boundaries,
                      i.e., left most and right most columns and top most and 
                      bottom most columns, otherwise the result may be incorrect.
  \param[in,out] dst  Destination image, should be the same cv::Mat::type() as
                      the source image.
  \param[in] mixGrad  True to apply mixing gradient operation.
*/
void PoissonImageEdit(const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst, bool mixGrad = false)
{
    CV_Assert(src.data && mask.data && dst.data);
    CV_Assert(src.size() == mask.size() && mask.size() == dst.size());
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    CV_Assert(dst.type() == src.type());
    CV_Assert(mask.type() == CV_8UC1);

    cv::Mat index;
    SparseMat A;
    cv::Mat b, x;
    int numElems;

    makeIndex(mask, index, numElems);
    if (src.type() == CV_8UC1)
    {
        getEquation(src, dst, mask, index, numElems, A, b, x, mixGrad);
        std::vector<int> diagPos;
        A.calcDiagonalElementsPositions(diagPos);
        solve(A.data, &A.count[0], &diagPos[0], (double*)b.data, (double*)x.data, 
            A.rows, A.maxCols, 10000, 0.01);
        copy(x, mask, index, dst);
    }
    else if (src.type() == CV_8UC3)
    {
        cv::Mat srcROISplit[3], dstROISplit[3];
        for (int i = 0; i < 3; i++)
        {
            srcROISplit[i].create(src.size(), CV_8UC1);
            dstROISplit[i].create(dst.size(), CV_8UC1);
        }
        cv::split(src, srcROISplit);
        cv::split(dst, dstROISplit);

        for (int i = 0; i < 3; i++)
        {
            getEquation(srcROISplit[i], dstROISplit[i], mask, index, numElems, 
                A, b, x, mixGrad);
            std::vector<int> diagPos;
            A.calcDiagonalElementsPositions(diagPos);
            solve(A.data, &A.count[0], &diagPos[0], (double*)b.data, (double*)x.data, 
                A.rows, A.maxCols, 10000, 0.01);
            copy(x, mask, index, dstROISplit[i]);
        }
        cv::merge(dstROISplit, 3, dst);
    }
}

/*!
  Overloaded Poisson image editing function.
  Source image, mask image and destination image should have the same size.
  Source image's content inside the mask's non zero region will be blended into 
  the destination image, using Possion image editing algorithm.
  \param[in] src          Source image, should be of type CV_8UC1 or CV_8UC3.
  \param[in] srcContour   A contour indicating the region of interest in the 
                          source image. Pixels inside the region will be blended to 
                          the destination image. The region enclosed by the contour 
                          should not contain pixels on the border of the source image.
  \param[in] ofsSrcToDst  The offset of the source image's region of interest in 
                          the destination image. Pixel (x, y) in the region of interest in 
                          the source image will be blended in (x, y) + ofsSrcToDst 
                          in the destination image. You should make sure 
                          that the destination image's region of interest locates totally inside 
                          the destionation image excluding the border pixels. 
  \param[in,out] dst      Destination image, should be the same cv::Mat::type() as
                          the source image.
  \param[in] mixGrad      True to apply mixing gradient operation.
*/
void PoissonImageEdit(const cv::Mat& src, const std::vector<cv::Point>& srcContour,
    cv::Point ofsSrcToDst, cv::Mat& dst, bool mixGrad = false)
{
    CV_Assert(src.data && (src.type() == CV_8UC1 || src.type() == CV_8UC3));
    CV_Assert(srcContour.size() >= 3);
    CV_Assert(dst.data && dst.type() == src.type());

    cv::Mat mask;
    cv::Rect srcRect;
    draw(srcContour, src.size(), srcRect, mask);

    cv::Mat srcROI = src(srcRect);
    cv::Mat dstROI = dst(srcRect + ofsSrcToDst);
    PoissonImageEdit(srcROI, mask, dstROI, mixGrad);
    return;
}

cv::Rect getNonZeroBoundingRectExtendOnePixel(const cv::Mat& mask)
{
    CV_Assert(mask.data && mask.type() == CV_8UC1);
    int rows = mask.rows, cols = mask.cols;
    int top = rows, bottom = -1, left = cols, right = -1;
    for (int i = 0; i < rows; i++)
    {
        if (cv::countNonZero(mask.row(i)))
        {
            top = i;
            break;
        }
    }
    for (int i = rows - 1; i >= 0; i--)
    {
        if (cv::countNonZero(mask.row(i)))
        {
            bottom = i;
            break;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        if (cv::countNonZero(mask.col(i)))
        {
            left = i;
            break;
        }
    }
    for (int i = cols - 1; i >= 0; i--)
    {
        if (cv::countNonZero(mask.col(i)))
        {
            right = i;
            break;
        }
    }
    CV_Assert(top > 0 && top < rows - 1 &&
        bottom > 0 && bottom < rows - 1 &&
        left > 0 && left < cols - 1 &&
        right > 0 && right < cols - 1);
    return cv::Rect(left - 1, top - 1, right - left + 3, bottom - top + 3);
}

/*!
  Overloaded Poisson image editing function.
  Source image, mask image and destination image should have the same size.
  Source image's content inside the mask's non zero region will be blended into 
  the destination image, using Possion image editing algorithm.
  \param[in] src          Source image, should be of type CV_8UC1 or CV_8UC3.
  \param[in] srcMask      Source image's mask indicating the region of interest in the 
                          source image. Pixels inside the region will be blended to 
                          the destination image. The region enclosed by the contour 
                          should not contain pixels on the border of the source image.
  \param[in] ofsSrcToDst  The offset of the source image's region of interest in 
                          the destination image. Pixel (x, y) in the region of interest in 
                          the source image will be blended in (x, y) + ofsSrcToDst 
                          in the destination image. You should make sure 
                          that the destination image's region of interest locates totally inside 
                          the destionation image excluding the border pixels. 
  \param[in,out] dst      Destination image, should be the same cv::Mat::type() as
                          the source image.
  \param[in] mixGrad      True to apply mixing gradient operation.
*/
void PoissonImageEdit(const cv::Mat& src, const cv::Mat& srcMask,
    cv::Point ofsSrcToDst, cv::Mat& dst, bool mixGrad = false)
{
    CV_Assert(src.data && (src.type() == CV_8UC1 || src.type() == CV_8UC3));
    CV_Assert(srcMask.data && srcMask.type() == CV_8UC1);
    CV_Assert(dst.data && dst.type() == src.type());

    cv::Rect srcRect = getNonZeroBoundingRectExtendOnePixel(srcMask);
    cv::Mat mask = srcMask(srcRect);

    cv::Mat srcROI = src(srcRect);
    cv::Mat dstROI = dst(srcRect + ofsSrcToDst);
    PoissonImageEdit(srcROI, mask, dstROI, mixGrad);
    return;
}

#define GRAY 1

void main()
{
    {
        // Images from http://www.ctralie.com/Teaching/PoissonImageEditing/
        cv::Mat src = cv::imread("GreatWhiteShark.jpg");
        cv::Mat dst = cv::imread("beach.jpg");
#if GRAY
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
#endif

        std::vector<cv::Point> contour(4);
        contour[0] = cv::Point(380, 300) - cv::Point(320, 230);
        contour[1] = cv::Point(550, 300) - cv::Point(320, 230);
        contour[2] = cv::Point(550, 420) - cv::Point(320, 230);
        contour[3] = cv::Point(380, 420) - cv::Point(320, 230);
        cv::Point ofsSrcToDst = cv::Point(320, 230);

        PoissonImageEdit(src, contour, ofsSrcToDst, dst, true);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::waitKey(0);
    }

    {
        // Eye photo and hand photo from 
        // https://en.wikipedia.org/wiki/Gradient-domain_image_processing
        cv::Mat src = cv::imread("220px-EyePhoto.jpg");
        cv::Mat dst = cv::imread("1074px-HandPhoto.jpg");
#if GRAY
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
#endif

        std::vector<cv::Point> srcContour(4);
        srcContour[0] = cv::Point(1, 1);
        srcContour[1] = cv::Point(218, 1);
        srcContour[2] = cv::Point(218, 130);
        srcContour[3] = cv::Point(1, 130);
        cv::Point ofsSrcToDst(570, 300);

        PoissonImageEdit(src, srcContour, ofsSrcToDst, dst, true);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::waitKey(0);
    }

    {
        // Eye photo from 
        // https://en.wikipedia.org/wiki/Gradient-domain_image_processing
        cv::Mat src = cv::imread("220px-EyePhoto.jpg");
        cv::Mat dst = cv::imread("1024px-Big_Tree_with_Red_Sky_in_the_Winter_Night.jpg");
#if GRAY
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
#endif

        std::vector<cv::Point> srcContour(4);
        srcContour[0] = cv::Point(1, 1);
        srcContour[1] = cv::Point(218, 1);
        srcContour[2] = cv::Point(218, 130);
        srcContour[3] = cv::Point(1, 130);
        cv::Point ofsSrcToDst(570, 300);

        PoissonImageEdit(src, srcContour, ofsSrcToDst, dst, true);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::waitKey(0);
    }

    // following image sources
    //  http://cs.brown.edu/courses/csci1950-g/results/proj2/pdoran/
    {
        cv::Mat src = cv::imread("src_img01.jpg");
        cv::Mat srcMask = cv::imread("mask_img01.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat dst = cv::imread("tar_img01.jpg");
#if GRAY
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
#endif
        cv::Point ofsSrcToDst(200, 200);
        cv::threshold(srcMask, srcMask, 128, 255, cv::THRESH_BINARY);
        PoissonImageEdit(src, srcMask, ofsSrcToDst, dst, true);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::waitKey(0);
    }

    {
        cv::Mat src = cv::imread("src_img02.jpg");
        cv::Mat srcMask = cv::imread("mask_img02.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat dst = cv::imread("tar_img02.jpg");
#if GRAY
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
#endif
        cv::Point ofsSrcToDst(20, 200);
        cv::threshold(srcMask, srcMask, 128, 255, cv::THRESH_BINARY);
        PoissonImageEdit(src, srcMask, ofsSrcToDst, dst, true);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::waitKey(0);
    }

    {
        cv::Mat src = cv::imread("src_img03.jpg");
        cv::Mat srcMask = cv::imread("mask_img03.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat dst = cv::imread("tar_img03.jpg");
#if GRAY
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
#endif
        cv::Point ofsSrcToDst(20, 20);
        cv::threshold(srcMask, srcMask, 128, 255, cv::THRESH_BINARY);
        PoissonImageEdit(src, srcMask, ofsSrcToDst, dst, true);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::waitKey(0);
    }

    {
        cv::Mat src = cv::imread("src_img04.jpg");
        cv::Mat srcMask = cv::imread("mask_img04.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat dst = cv::imread("tar_img04.jpg");
#if GRAY
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
        cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
#endif
        cv::Point ofsSrcToDst(20, 20);
        cv::threshold(srcMask, srcMask, 128, 255, cv::THRESH_BINARY);
        PoissonImageEdit(src, srcMask, ofsSrcToDst, dst, true);
        cv::imshow("src", src);
        cv::imshow("dst", dst);
        cv::waitKey(0);
    }
    
}