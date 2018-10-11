#include "fisheye.hpp"

/************************************
* Method:     distortPoints
* Brief:      对点进行畸变
* Author:     朱凌峰
* Date:       2018/05/13
* Description:对opencv的cv::fisheye::distortPoints
              的封装，也是重载, 参数的含义同cv::fisheye::undistortPoints
* Returns:    void
* Parameter:  undistorted  输入的无畸变的点
Array of object points, 1xN/Nx1 2-channel (or vector<Point2f> ), where N is the number of points in the view.
* Parameter:  distorted    输出的畸变后的点
Output array of image points, 1xN/Nx1 2-channel, or vector<Point2f> .
* Parameter:  K            相机内参
* Parameter:  D			   相机畸变参数
Input vector of distortion coefficients (k1,k2,k3,k4).
* Parameter:  R			   相机外参
Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3 1-channel or 1x1 3-channel
* Parameter:  P			   图像畸变矫正时采用的新相机内参
New camera matrix (3x3) or new projection matrix (3x4)
************************************/
void distortPoints(InputArray undistorted, OutputArray distorted,
	InputArray K, InputArray D, InputArray R, InputArray P) {
	/*************copy cv::fisheye::undistortPoints部分***********/
	//输入数据处理
	CV_Assert(undistorted.type() == CV_32FC2 || undistorted.type() == CV_64FC2);
	CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));
	CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
	CV_Assert(D.total() == 4 && K.size() == Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

	distorted.create(undistorted.size(), undistorted.type());
	int N = undistorted.total();//点的数目

	cv::Matx33d RR = cv::Matx33d::eye();
	if (!R.empty() && R.total() * R.channels() == 3)
	{
		cv::Vec3d rvec;
		R.getMat().convertTo(rvec, CV_64F);
		RR = Affine3d(rvec).rotation();
	}
	else if (!R.empty() && R.size() == Size(3, 3))
		R.getMat().convertTo(RR, CV_64F);

	cv::Matx33d PP = cv::Matx33d::eye();
	if (!P.empty())
		P.getMat().colRange(0, 3).convertTo(PP, CV_64F);

	cv::Matx33d iR = (PP * RR).inv(cv::DECOMP_SVD);//从图像坐标到世界坐标（R为单位矩阵时，相机坐标）的映射

	const cv::Vec2f* srcf = undistorted.getMat().ptr<cv::Vec2f>();
	const cv::Vec2d* srcd = undistorted.getMat().ptr<cv::Vec2d>();
	int sdepth = undistorted.depth();
	/*************copy cv::fisheye::undistortPoints部分——结束***********/

	Mat p_und = Mat(3, N, CV_64FC1);//齐次坐标的无畸变点(3*N)
	for (int i = 0; i < N; i++) {
		Vec2d temp = (sdepth == CV_32F ? (Vec2d)srcf[i] : srcd[i]);
		p_und.at<double>(0, i) = temp[0];
		p_und.at<double>(1, i) = temp[1];
	}
	//使用split方法失败，原因未知
	//vector<Mat> channels;
	//split(undistorted, channels);
	//cout << "channels[0]:" << format(channels[0], Formatter::FMT_NUMPY) << endl;
	//p_und.row(0) = channels[0];
	//p_und.row(1) = channels[1];
	p_und.row(2) = Mat::ones(1, N, CV_64FC1);
	//cout << "p_und:" << format(p_und, Formatter::FMT_NUMPY) << endl;

	Mat Pcn(3, N, CV_64FC1);//归一化坐标
	Mat Min_new_inv = P.getMat().inv();
	Pcn = Min_new_inv*p_und;
	//cout << "Pcn:" << format(Pcn, Formatter::FMT_NUMPY) << endl;

	Mat PcnC2 = Mat(1, N, CV_64FC2);//双通道的归一化坐标
	vector<Mat> temp;
	temp.push_back(Pcn.row(0));
	temp.push_back(Pcn.row(1));
	merge(temp, PcnC2);
	//cout << "PcnC2:" << format(PcnC2, Formatter::FMT_NUMPY) << endl;

	fisheye::distortPoints(PcnC2, distorted, K, D);
	//cout << "distorted:" << format(distorted, Formatter::FMT_NUMPY) << endl;
}

/************************************
* Method:     myFisheyeCalib
* Brief:      鱼眼相机标定函数
* Author:     朱凌峰
* Date:       2018/05/13
* Description:完整的鱼眼相机标定流程，包括图片载入、亚像素角点提取、标定、标定
结果评价、标定结果保存到文件。相关文件都要在工程目录下。
* Returns:    void
* Parameter:  calibImgFolder 标定图片目录（图片需是BMP文件）
* Parameter:  cameraParamsFileName 相机参数保存文件名（ymal)文件
************************************/
//void myFisheyeCalib(string calibImgFolder, string cameraParamsFileName) {
//	ofstream fout(calibImgFolder + "_caliberation_result.txt");  /**    保存定标结果的文件     **/
//	/************************************************************************
//	读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
//	*************************************************************************/
//	cout << "开始提取角点………………" << endl;

//	string dir = calibImgFolder;
//	_chdir(dir.data());
//	intptr_t hFile;
//	_finddata_t fileinfo;

//	//int image_count = 61;                    /****    图像数量     ****/
//	Size board_size = Size(9, 6);            /****    定标板上每行、列的角点数       ****/
//	vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/
//	vector<vector<Point2f>>  corners_Seq;    /****  保存检测到的所有角点       ****/
//	vector<Mat>  image_Seq;
//	vector<string>  imagename_Seq;
//	int successImageNum = 0;				/****	成功提取角点的棋盘图数量	****/
//	
//	int count = 0;		/*count是什么？*/
//	if ((hFile = _findfirst("*.bmp", &fileinfo)) == -1L) {
//		cout << "No BMP file in " << dir << endl;
//	}
//	else {
//		do {
//			string imageFileName;
//			cout << "Frame # " << imageFileName << " ..." << endl;
//			cv::Mat imageColor = imread(imageFileName.append(fileinfo.name));
//			//resize(imageColor, imageColor, outsSize, 0, 0, INTER_AREA);
//			/* 提取角点 */
//			Mat imageGray;
//			cvtColor(imageColor, imageGray, CV_RGB2GRAY);

//			//Mat imageColorExtend, imageGrayExtend;
//			//copyMakeBorder(imageColor, imageColorExtend, 240, 240, 320, 320, BORDER_CONSTANT);
//			//copyMakeBorder(imageGray, imageGrayExtend, 240, 240, 320, 320, BORDER_CONSTANT);

//			//imshow("imageColor", imageColor);
//			//imshow("imageGray", imageGray);
//			//imshow("imageColorExtend", imageColorExtend);
//			//imshow("imageGrayExtend", imageGrayExtend);
//			//waitKey(0);

//			bool patternfound = findChessboardCorners(imageColor, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
//			if (!patternfound)
//			{
//				cout << "can not find chessboard corners!\n";
//				continue;
//				//exit(1);
//			}
//			else
//			{
//				/* 亚像素精确化 */
//				cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
//				/* 绘制检测到的角点并保存 */
//				Mat imageTemp = imageColor.clone();
//				for (int j = 0; j < corners.size(); j++)
//				{
//					circle(imageTemp, corners[j], 10, Scalar(0, 0, 255), 2, 8, 0);
//				}
//				imwrite(imageFileName + "_corner.jpg", imageTemp);
//				cout << "Frame corner# " << imageFileName << " ...end" << endl;

//				count = count + corners.size();
//				successImageNum = successImageNum + 1;
//				corners_Seq.push_back(corners);
//				imagename_Seq.push_back(imageFileName);
//			}
//			image_Seq.push_back(imageColor);
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//		_chdir("../");
//	}

//	cout << "角点提取完成！\n" << endl;
//	/************************************************************************
//	摄像机定标
//	*************************************************************************/
//	cout << "开始定标………………" << endl;
//	Size square_size = Size(60, 60);			/*棋盘格尺寸*/
//	vector<vector<Point3f>>  object_Points;        /****  保存定标板上角点的三维坐标   ****/

//	Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));  /*****   保存提取的所有角点   *****/
//	vector<int>  point_counts;
//	/* 初始化定标板上角点的三维坐标 */
//	for (int t = 0; t < successImageNum; t++)
//	{
//		vector<Point3f> tempPointSet;
//		for (int i = 0; i < board_size.height; i++)
//		{
//			for (int j = 0; j < board_size.width; j++)
//			{
//				/* 假设定标板放在世界坐标系中z=0的平面上 */
//				Point3f tempPoint;
//				tempPoint.x = i*square_size.width;
//				tempPoint.y = j*square_size.height;
//				tempPoint.z = 0;
//				tempPointSet.push_back(tempPoint);
//				//cout << tempPoint.x << tempPoint.y << endl;
//			}
//		}
//		object_Points.push_back(tempPointSet);
//	}
//	for (int i = 0; i < successImageNum; i++)
//	{
//		point_counts.push_back(board_size.width*board_size.height);
//	}
//	/* 开始定标 */
//	Size image_size = image_Seq[0].size();

//	cv::Matx33d intrinsic_matrix;    /*****    摄像机内参数矩阵    ****/
//	cv::Vec4d distortion_coeffs;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
//	std::vector<cv::Vec3d> rotation_vectors;                           /* 每幅图像的旋转向量 */
//	std::vector<cv::Vec3d> translation_vectors;                        /* 每幅图像的平移向量 */
//	int flags = 0;
//	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
//	flags |= cv::fisheye::CALIB_CHECK_COND;
//	flags |= cv::fisheye::CALIB_FIX_SKEW;

//	fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 500, 1e-10));
//	cout << "定标完成！\n" << endl;

//	/************************************************************************
//	对定标结果进行评价
//	*************************************************************************/
//	cout << "开始评价定标结果………………" << endl;
//	double total_err = 0.0;                   /* 所有图像的平均误差的总和 */
//	double err = 0.0;                        /* 每幅图像的平均误差 */
//	vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/

//	cout << "每幅图像的定标误差：" << endl;
//	cout << "每幅图像的定标误差：" << endl << endl;
//	for (int i = 0; i < corners_Seq.size(); i++)
//	{
//		vector<Point3f> tempPointSet = object_Points[i];
//		/****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
//		fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
//		/* 计算新的投影点和旧的投影点之间的误差*/
//		vector<Point2f> tempImagePoint = corners_Seq[i];
//		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
//		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
//		for (size_t i = 0; i != tempImagePoint.size(); i++)
//		{
//			image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
//			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
//			//cout << tempImagePoint[i].y;
//			//cout << image_points2[i].y;
//		}
//		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
//		total_err += err /= point_counts[i];
//		cout << imagename_Seq[i] << "幅图像的平均误差：" << err << "像素" << endl;
//		fout << imagename_Seq[i] << "幅图像的平均误差：" << err << "像素" << endl;
//	}
//	cout << "总体平均误差：" << total_err / corners_Seq.size() << "像素" << endl;
//	fout << "总体平均误差：" << total_err / corners_Seq.size() << "像素" << endl << endl;
//	cout << "评价完成！" << endl << endl;

//	/************************************************************************
//	保存定标结果
//	*************************************************************************/
//	//初始化
//	cout << "保存定标结果………………" << endl;
//	FileStorage fs(cameraParamsFileName, FileStorage::WRITE);

//	//开始文件写入
//	fs << "frameCount" << successImageNum;
//	time_t rawtime; time(&rawtime);
//	fs << "calibrationDate" << asctime(localtime(&rawtime));
//	Mat cameraMatrix = Mat(intrinsic_matrix);
//	Mat distCoeffs = Mat(distortion_coeffs);
//	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;

//	//fs << "cameraMatrix" << intrinsic_matrix << "distCoeffs" << distortion_coeffs;
//	fs.release();
//	cout << "保存定标结果完成！" << endl << endl;
//	if (0) {
//		//cout << "开始保存定标结果………………" << endl;
//		//Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */

//		//fout << "相机内参数矩阵：" << endl;
//		//fout << intrinsic_matrix << endl;
//		//fout << "畸变系数：\n";
//		//fout << distortion_coeffs << endl;
//		//for (int i = 0; i < corners_Seq.size(); i++)
//		//{
//		//	fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
//		//	fout << rotation_vectors[i] << endl;

//		//	/* 将旋转向量转换为相对应的旋转矩阵 */
//		//	Rodrigues(rotation_vectors[i], rotation_matrix);
//		//	fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
//		//	fout << rotation_matrix << endl;
//		//	fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
//		//	fout << translation_vectors[i] << endl;
//		//}
//		//cout << "完成保存" << endl << endl;
//		//fout << endl;
//		//fout.close();
//		//FileStorage fs1("003intrinsic_matrix.xml", FileStorage::APPEND);
//		//fs1 << direction << Mat(intrinsic_matrix);
//		//fs1.release();

//		//FileStorage fs2("003distortion_coeffs.xml", FileStorage::APPEND);
//		//fs2 << direction << Mat(distortion_coeffs);
//		//fs2.release();
//	}
//}

void testDistortPoints(){
	string cameraParamsFileName = "cameraParams.yaml";//相机参数保存文件名（ymal文件）
	string calibImgFolder = "FRONT";//标定图片目录	
	//相机标定
	//myFisheyeCalib(calibImgFolder, cameraParamsFileName);
	/************************************************************************
	读取相机参数
	*************************************************************************/
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
	图像畸变矫正
	*************************************************************************/
	cout << "Undistort image 图像畸变矫正............" << endl;

	/***确定畸变矫正的映射关系****/
	Size size_out1;  /*畸变矫正图像尺寸*/
	size_out1.width = 1280;
	size_out1.height = 1080;
	Mat mapx = Mat(size_out1, CV_32FC1);
	Mat mapy = Mat(size_out1, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	/*获取新的相机矩阵*/
	Mat newCameraMatrix = Mat(intrinsic_matrix).clone();
	double balance = 1;//在最大焦距和最小焦距间插值
	double fov_scale = 0.3;//视野缩放比例
	fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix, distortion_coeffs,
		size_out1, R, newCameraMatrix, balance, size_out1, fov_scale);

	/*尝试用单位阵作为newCameraMatrix*/
	//Mat identity = Mat::eye(3, 3, CV_32F);
	//newCameraMatrix= identity.clone();

	/*获取畸变矫正映射*/
	fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, newCameraMatrix, size_out1, CV_32FC1, mapx, mapy);

	/*对图像进行畸变矫正*/
	string imgFolder = "resources/";
	string imgFileName = "F1_5.jpg";
	Mat fisheyeImg = imread(imgFolder + imgFileName);
	cout<<"fisheyeImg:"<<imgFolder + imgFileName<<endl;
	if (fisheyeImg.empty()) {
		cout << "Can not open image file 无法打开图像文件\n";
		return ;
	}
	Mat undistortImg;
	remap(fisheyeImg, undistortImg, mapx, mapy, INTER_LINEAR);

	imwrite(imgFileName + "resources/F1_5_Undistort_1_0.3.jpg", undistortImg);
	cout << "Undistort image  done畸变矫正完成！" << endl << endl;
	if (0) {
		/************************************************************************
		对无畸变的点进行畸变
		*************************************************************************/
		//cout << "对无畸变的点进行畸变..." << endl;
		///*无畸变的点初始化*/
		//int x1 = 780, y1 = 298, x2 = 1058, y2 = 393;//(c,r)
		//Point2d undistortPoint1(x1, y1);//无畸变的点
		//Point2d undistortPoint2(x2, y2);
		//Mat undistorted = Mat(1, 2, CV_64FC2);//无畸变的点
		//undistorted.at<Vec2d>(0, 0) = undistortPoint1;
		//undistorted.at<Vec2d>(0, 1) = undistortPoint2;
		//cout << "undistorted:" << format(undistorted, Formatter::FMT_NUMPY) << endl;

		//Mat distorted;//畸变后的点
		//distortPoints(undistorted, distorted, intrinsic_matrix, distortion_coeffs,
		//	R,newCameraMatrix);

		//Point distortPoint1(distorted.at<Vec2d>(0, 0));//畸变后的点
		//Point distortPoint2(distorted.at<Vec2d>(0, 1));
		//cout << "distorted:" << format(distorted, Formatter::FMT_NUMPY) << endl;
		//
		///*图像显示*/
		//line(undistortImg, undistortPoint1, undistortPoint2, Scalar(255, 255, 255), 5);
		//namedWindow("畸变矫正图像", WINDOW_NORMAL);
		//imshow("畸变矫正图像", undistortImg);

		//line(fisheyeImg, distortPoint1, distortPoint2, Scalar(255, 255,255), 5);
		//namedWindow("鱼眼图像", WINDOW_NORMAL);
		//imshow("鱼眼图像", fisheyeImg);
		//
		///*用opencv的undistortPoints函数验证我的distortPoints函数的正确性*/
		//Mat undistortedForSure;
		//fisheye::undistortPoints(distorted, undistortedForSure, intrinsic_matrix, distortion_coeffs,
		//	R, newCameraMatrix);
		//cout << "undistortedForSure:" << format(undistortedForSure, Formatter::FMT_NUMPY) << endl;

		//cout << "对无畸变的点进行畸变完成!" << endl<<endl;
	}
	/************************************************************************
	对无畸变的矩形四个顶点进行畸变
	*************************************************************************/
	cout << "Distort four points 对无畸变的矩形四个顶点进行畸变..." << endl;
	/*无畸变的点初始化*/
	//int x1 = 780, y1 = 298, x2 = 1058, y2 = 393;//(c,r)
	int x1 = 816, y1 = 279, x2 = 1214, y2 = 402;//(c,r)

	Point2d undistortPoint1(x1, y1);//无畸变的点
	Point2d undistortPoint2(x2, y1);
	Point2d undistortPoint3(x2, y2);
	Point2d undistortPoint4(x1, y2);
	Mat undistorted = Mat(1, 4, CV_64FC2);//无畸变的点
	undistorted.at<Vec2d>(0, 0) = undistortPoint1;
	undistorted.at<Vec2d>(0, 1) = undistortPoint2;
	undistorted.at<Vec2d>(0, 2) = undistortPoint3;
	undistorted.at<Vec2d>(0, 3) = undistortPoint4;
	cout << "undistorted:" << format(undistorted, Formatter::FMT_NUMPY) << endl;

	Mat distorted;//畸变后的点
	distortPoints(undistorted, distorted, intrinsic_matrix, distortion_coeffs,
		R, newCameraMatrix);

	Point distortPoint1(distorted.at<Vec2d>(0, 0));//畸变后的点
	Point distortPoint2(distorted.at<Vec2d>(0, 1));
	Point distortPoint3(distorted.at<Vec2d>(0, 2));
	Point distortPoint4(distorted.at<Vec2d>(0, 3));
	cout << "distorted:" << format(distorted, Formatter::FMT_NUMPY) << endl;

	/*图像显示*/
	line(undistortImg, undistortPoint1, undistortPoint2, Scalar(255, 255, 255), 2);
	line(undistortImg, undistortPoint2, undistortPoint3, Scalar(255, 255, 255), 2);
	line(undistortImg, undistortPoint3, undistortPoint4, Scalar(255, 255, 255), 2);
	line(undistortImg, undistortPoint4, undistortPoint1, Scalar(255, 255, 255), 2);
	namedWindow("畸变矫正图像", WINDOW_NORMAL);
	imshow("畸变矫正图像", undistortImg);
	imwrite("资源/畸变矫正图像及物体矩形框.jpg", undistortImg);
	line(fisheyeImg, distortPoint1, distortPoint2, Scalar(255, 255, 255), 2);
	line(fisheyeImg, distortPoint2, distortPoint3, Scalar(255, 255, 255), 2);
	line(fisheyeImg, distortPoint3, distortPoint4, Scalar(255, 255, 255), 2);
	line(fisheyeImg, distortPoint4, distortPoint1, Scalar(255, 255, 255), 2);
	namedWindow("鱼眼图像", WINDOW_NORMAL);
	imshow("鱼眼图像", fisheyeImg);
	imwrite("资源/鱼眼图像图像及物体矩形框.jpg", fisheyeImg);

	cout << "Distort four points done 对无畸变的矩形四个顶点进行畸变完成!" << endl << endl;
}



