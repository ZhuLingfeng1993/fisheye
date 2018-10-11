#include "../fisheye.hpp"

void batchUndistortImgs(){
	/************************************************************************
	获取相机参数
	*************************************************************************/
	string cameraParamsFileName = "cameraParams960p.yaml";//相机参数保存文件名（ymal文件）
	string calibImgFolder = "FRONT";//标定图片目录	
	//相机标定
	//myFisheyeCalib(calibImgFolder, cameraParamsFileName);

	cout << "Reading camera parameters 读取相机参数............" << endl;
	//初始化
	FileStorage fs2(cameraParamsFileName, FileStorage::READ);

	// 第一种方法，对FileNode操作
	int frameCount = (int)fs2["frameCount"];

	std::string date;
	// 第二种方法，使用FileNode运算符> > 
	fs2["calibrationDate"] >> date;

	Mat intrinsic_matrix, distortion_coeffs;
	fs2["cameraMatrix"] >> intrinsic_matrix;
	fs2["distCoeffs"] >> distortion_coeffs;

	cout << "frameCount: " << frameCount << endl
		<< "calibration date: " << date << endl
		<< "camera matrix: " << intrinsic_matrix << endl
		<< "distortion coeffs: " << distortion_coeffs << endl;

	fs2.release();
	cout << "Reading camera parameters done 读取相机参数完成！" << endl << endl;

	/************************************************************************
	确定畸变矫正的映射关系
	*************************************************************************/
	Size size_in;  /*鱼眼图像尺寸*/
	size_in.width = 1280;
	size_in.height = 960;
	Size size_out;  /*畸变矫正图像尺寸*/
	size_out.width = size_in.width/1;
	size_out.height = size_in.height/1;
	Mat mapx = Mat(size_out, CV_32FC1);
	Mat mapy = Mat(size_out, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	/*获取新的相机矩阵*/
	Mat newCameraMatrix = Mat(intrinsic_matrix).clone();
	double balance = 1;//在最大焦距和最小焦距间插值
	double fov_scale = 0.5;//视野缩放比例
	fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix, distortion_coeffs,
		size_in, R, newCameraMatrix, balance, size_out, fov_scale);
	/*获取畸变矫正映射*/
	fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, newCameraMatrix, size_out, CV_32FC1, mapx, mapy);

	/************************************************************************
	对图像进行畸变矫正
	*************************************************************************/
	cout << "Batch undistort image...... 图片批量畸变矫正！" << endl << endl;
	bool testImgFlag = true;//是否进行测试的标志位
	
	string imgFolder = "/media/zhulingfeng/LENOVO/zlf_od_Ubuntu_backup/data/newData2";
	if( testImgFlag == true)
		imgFolder = "./testImg";

	int i = 0;
	int imageNum=1127;
	if( testImgFlag == true)
		imageNum = 10;
	while (i < imageNum)
	{
		Mat fisheyeImg = imread(imgFolder + "/many_people_" + std::to_string(i) + ".jpg");
		if (fisheyeImg.empty()) {
			cout << "Can not open image file 无法打开图像文件\n"<<(imgFolder + "/many_people_" + std::to_string(i) + ".jpg")<<endl;
			return ;
		}
		Mat undistortImg;
		cv::remap(fisheyeImg, undistortImg, mapx, mapy, INTER_LINEAR);
		string imout_name = "/media/zhulingfeng/LENOVO/zlf_od_Ubuntu_backup/data/newData2Undistort/many_people_" + std::to_string(i) + "Undistort" + ".jpg";
		if( testImgFlag == true)
			imout_name = "./testImgUndistort/many_people_" + std::to_string(i) + "Undistort" + ".jpg";
//		cout<<imout_name<<endl;
		imwrite(imout_name, undistortImg);
		i = i + 1;
		if(i%100==0)
			cout<<i/100*100<<" images processed."<<endl;
	}
	cout << "Batch undistort image  done! 图片批量畸变矫正完成！" << endl << endl;
}

int main()
{
	batchUndistortImgs();
	waitKey(0);
	return 0;
}
