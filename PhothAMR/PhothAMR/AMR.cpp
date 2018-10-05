// svm.cpp : �������̨Ӧ�ó������ڵ㡣
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

void on_mouse(int event, int x, int y, int flags, void* ustc); //���ȡ������
void getTrainData(Mat &train_data, Mat &train_label);  //����ѵ������ 
void svm(); //svm����


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

//�����ͼ����ȡ�����㣬��q���˳�
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Point pt = Point(x, y);
		Vec3b point = img.at<Vec3b>(y, x);  //ȡ�������괦������ֵ��ע��x,y��˳��
		Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
		if (flag)
		{
			targetData.push_back(tmp); //��������������
			circle(img, pt, 2, Scalar(0, 255, 255), -1, 8); //��������ĵ� 

		}

		else
		{
			backData.push_back(tmp); //���븺��������
			circle(img, pt, 2, Scalar(255, 0, 0), -1, 8);

		}
		imshow(wdname, img);
	}
}

void getTrainData(Mat &train_data, Mat &train_label)
{
	int m = targetData.rows;
	int n = backData.rows;
	cout << "����������:" << m << endl;
	cout << "����������" << n << endl;
	vconcat(targetData, backData, train_data); //�ϲ����е������㣬��Ϊѵ������
	train_label = Mat(m + n, 1, CV_32S, Scalar::all(1)); //��ʼ����ע
	for (int i = m; i < m + n; i++)
		train_label.at<int>(i, 0) = -1;
}

void svm()
{
	Mat train_data, train_label;
	getTrainData(train_data, train_label); //��ȡ���ѡ�������ѵ������

										   // ���ò���
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);

	// ѵ��������
	Ptr<TrainData> tData = TrainData::create(train_data, ROW_SAMPLE, train_label);
	svm->train(tData);

	Vec3b color(0, 0, 0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Vec3b point = img.at<Vec3b>(i, j);  //ȡ�������괦������ֵ
			Mat sampleMat = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
			float response = svm->predict(sampleMat);  //����Ԥ�⣬����1��-1,��������Ϊfloat
			if ((int)response != 1)
				image.at<Vec3b>(i, j) = color;  //����������Ϊ��ɫ
		}

	imshow("SVM Simple Example", image);
	imwrite("amr.jpg", image);
	waitKey(0);
}