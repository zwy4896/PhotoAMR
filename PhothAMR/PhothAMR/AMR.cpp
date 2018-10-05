// svm.cpp : 定义控制台应用程序的入口点。
//

#include<iostream>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace cv::ml;
using namespace std;

Mat img, image;
Mat targetData, backData;
bool flag = true;
string wdname = "image";

void on_mouse(int event, int x, int y, int flags, void* ustc); //鼠标取样本点
void getTrainData(Mat &train_data, Mat &train_label);  //生成训练数据 
void svm(); //svm分类


int main(int argc, char** argv)
{
	string path = "./IMG_4181.jpg";
	img = imread(path);
	img.copyTo(image);
	if (img.empty())
	{
		cout << "Image load error";
		return 0;
	}
	namedWindow(wdname);
	setMouseCallback(wdname, on_mouse, 0);

	for (;;)
	{
		imshow("image", img);

		int c = waitKey(0);
		if ((c & 255) == 27)
		{
			cout << "Exiting ...\n";
			break;
		}
		if ((char)c == 'c')
		{
			flag = !flag;
		}
		if ((char)c == 'q')
		{
			destroyAllWindows();
			break;
		}
	}
	svm();
	return 0;
}

//鼠标在图像上取样本点，按q键退出
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Point pt = Point(x, y);
		Vec3b point = img.at<Vec3b>(y, x);  //取出该坐标处的像素值，注意x,y的顺序
		Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
		if (flag)
		{
			targetData.push_back(tmp); //加入正样本矩阵
			circle(img, pt, 2, Scalar(0, 255, 255), -1, 8); //画出点击的点 

		}

		else
		{
			backData.push_back(tmp); //加入负样本矩阵
			circle(img, pt, 2, Scalar(255, 0, 0), -1, 8);

		}
		imshow(wdname, img);
	}
}

void getTrainData(Mat &train_data, Mat &train_label)
{
	int m = targetData.rows;
	int n = backData.rows;
	cout << "正样本数：:" << m << endl;
	cout << "负样本数：" << n << endl;
	vconcat(targetData, backData, train_data); //合并所有的样本点，作为训练数据
	train_label = Mat(m + n, 1, CV_32S, Scalar::all(1)); //初始化标注
	for (int i = m; i < m + n; i++)
		train_label.at<int>(i, 0) = -1;
}

void svm()
{
	Mat train_data, train_label;
	getTrainData(train_data, train_label); //获取鼠标选择的样本训练数据

										   // 设置参数
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);

	// 训练分类器
	Ptr<TrainData> tData = TrainData::create(train_data, ROW_SAMPLE, train_label);
	svm->train(tData);

	Vec3b color(0, 0, 0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Vec3b point = img.at<Vec3b>(i, j);  //取出该坐标处的像素值
			Mat sampleMat = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
			float response = svm->predict(sampleMat);  //进行预测，返回1或-1,返回类型为float
			if ((int)response != 1)
				image.at<Vec3b>(i, j) = color;  //将背景设置为黑色
		}

	imshow("SVM Simple Example", image);
	imwrite("amr.jpg", image);
	waitKey(0);
}