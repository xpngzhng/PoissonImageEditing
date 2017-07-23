#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

template <typename Type>
void solve(const Type* A, const Type* b, Type* x,
    int size, int stepInBytes, int maxIters, Type eps)
{
    for (int iter = 0; iter < maxIters; iter++)
    {
        int count = 0;
        for (int i = 0; i < size; i++)
        {
            Type val = 0;
            const Type* ptrRow = A + stepInBytes / sizeof(Type) * i;
            for (int j = 0; j < i; j++)
                val += ptrRow[j] * x[j];
            for (int j = i + 1; j < size; j++)
                val += ptrRow[j] * x[j];
            val = b[i] - val;
            val /= ptrRow[i];
            if (fabs(val - x[i]) < eps)
                count++;
            x[i] = val;
        }
        if (count == size)
        {
            printf("count = %d, end\n", iter + 1);
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

void draw(const std::vector<cv::Point>& contour, cv::Size& imageSize, cv::Rect& extendRect, cv::Mat& mask)
{
    cv::Rect contourRect = cv::boundingRect(contour);
    int left, right, top, bottom;
    left = contourRect.x;
    right = contourRect.x + contourRect.width;
    top = contourRect.y;
    bottom = contourRect.y + contourRect.height;
    if (left > 0) left--;
    if (right < imageSize.width) right++;
    if (top > 0) top--;
    if (bottom < imageSize.height) bottom++;
    extendRect.x = left;
    extendRect.y = top;
    extendRect.width = right - left;
    extendRect.height = bottom - top;
    mask.create(extendRect.height, extendRect.width, CV_8UC1);
    mask.setTo(0);
    std::vector<std::vector<cv::Point> > contours(1);
    contours[0] = contour;
    cv::drawContours(mask, contours, -1, cv::Scalar(255), -1, 8, cv::noArray(), 0, cv::Point(-left, -top));
}

void getEquation(const cv::Mat& src, const cv::Mat& dst, 
    const cv::Mat& mask, const cv::Mat& index, int count,
    cv::Mat& A, cv::Mat& b, cv::Mat& x)
{
    CV_Assert(src.data && dst.data && mask.data && index.data);
    CV_Assert((src.type() == CV_8UC1) && (dst.type() == CV_8UC1) &&
        (mask.type() == CV_8UC1) && (index.type() == CV_32SC1));
    CV_Assert((src.size() == dst.size()) && (src.size() == mask.size()) && (src.size() == index.size()));
    
    int rows = src.rows, cols = src.cols;
    A.create(count, count, CV_64FC1);
    A.setTo(0);
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
                int neighborCount = 0;
                int bVal = 0;
                if (i > 0)
                {
                    neighborCount++;
                    if (mask.at<unsigned char>(i - 1, j))
                    {
                        int topIndex = index.at<int>(i - 1, j);
                        A.at<double>(currIndex, topIndex) = -1;
                    }
                    else
                    {
                        bVal += dst.at<unsigned char>(i - 1, j);
                    }
                    bVal += (currSrcVal - src.at<unsigned char>(i - 1, j));
                }
                if (i < rows - 1)
                {
                    neighborCount++;
                    if (mask.at<unsigned char>(i + 1, j))
                    {
                        int bottomIndex = index.at<int>(i + 1, j);
                        A.at<double>(currIndex, bottomIndex) = -1;
                    }
                    else
                    {
                        bVal += dst.at<unsigned char>(i + 1, j);
                    }
                    bVal += (currSrcVal - src.at<unsigned char>(i + 1, j));
                }
                if (j > 0)
                {
                    neighborCount++;
                    if (mask.at<unsigned char>(i, j - 1))
                    {
                        int leftIndex = index.at<int>(i, j - 1);
                        A.at<double>(currIndex, leftIndex) = -1;
                    }
                    else
                    {
                        bVal += dst.at<unsigned char>(i, j - 1);
                    }
                    bVal += (currSrcVal - src.at<unsigned char>(i, j - 1));
                }
                if (j < cols - 1)
                {
                    neighborCount++;
                    if (mask.at<unsigned char>(i, j + 1))
                    {
                        int rightIndex = index.at<int>(i, j + 1);
                        A.at<double>(currIndex, rightIndex) = -1;
                    }
                    else
                    {
                        bVal += dst.at<unsigned char>(i, j + 1);
                    }
                    bVal += (currSrcVal - src.at<unsigned char>(i, j + 1));
                }
                A.at<double>(currIndex, currIndex) = neighborCount;
                b.at<double>(currIndex) = bVal;
                //x.at<double>(currIndex) = currSrcVal;
                x.at<double>(currIndex) = dst.at<unsigned char>(i, j);
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

void print(const cv::Mat& mat)
{
    for (int j = 0; j < mat.cols; j++)
        printf("%6d ", j);
    printf("\n");
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            printf("%6.2f ", mat.at<double>(i, j));
        }
        printf("\n");
    }
}

void main()
{
    //double A[] = {10, -1, 2, 0,
    //             -1, 11, -1, 3,
    //             2, -1, 10, -1,
    //             0, 3, -1, 8};
    //double b[] = {6, 25, -11, 15};
    //double x[] = {0, 0, 0, 0};
    //double r[] = {0, 0, 0, 0};
    //solve(A, b, r, 4, 4 * sizeof(double), 1000, 0.0001);
    //return;

    //std::vector<std::vector<cv::Point> > contours(1);
    //contours[0].resize(4);
    //contours[0][0] = cv::Point(10, 10);
    //contours[0][1] = cv::Point(10, 30);
    //contours[0][2] = cv::Point(30, 30);
    //contours[0][3] = cv::Point(30, 10);
    //cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC1);
    //cv::drawContours(image, contours, -1, cv::Scalar(255), -1, 8, cv::noArray(), 0, cv::Point(-10, -10));
    //cv::imshow("image", image);
    //cv::waitKey(0);

    cv::Mat srcColor = cv::imread("220px-EyePhoto.jpg");
    cv::Mat dstColor = cv::imread("1074px-HandPhoto.jpg");

    cv::Mat src, dst;
    cv::cvtColor(srcColor, src, CV_BGR2GRAY);
    cv::cvtColor(dstColor, dst, CV_BGR2GRAY);

    std::vector<cv::Point> srcContour(4);
    //srcContour[0] = cv::Point(1, 1);
    //srcContour[1] = cv::Point(218, 1);
    //srcContour[2] = cv::Point(218, 130);
    //srcContour[3] = cv::Point(1, 130);
    srcContour[0] = cv::Point(1, 1);
    srcContour[1] = cv::Point(58, 1);
    srcContour[2] = cv::Point(58, 60);
    srcContour[3] = cv::Point(1, 60);
    cv::Point ofsSrcToDst(570, 300);

    cv::Rect extendRect;
    cv::Mat mask, index;
    cv::Mat A, b, x;
    int numElems;

    draw(srcContour, src.size(), extendRect, mask);
    cv::imshow("mask", mask);
    cv::waitKey(0);
    makeIndex(mask, index, numElems);

    cv::Mat srcROI(src, extendRect), dstROI(dst, extendRect + ofsSrcToDst);
    getEquation(srcROI, dstROI, mask, index, numElems, A, b, x);
    //print(A);
    //return;
    solve((double*)A.data, (double*)b.data, (double*)x.data, numElems, A.step[0], 10000, 0.01);
    copy(x, mask, index, dstROI);    

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    //cv::imwrite("dst.bmp", dst);
}