#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

using namespace cv;
using namespace std;

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        Mat _src(src.cols(), src.rows(), DataType<_Tp>::type,
              (void*)src.data(), src.stride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        Mat _src(src.rows(), src.cols(), DataType<_Tp>::type,
                 (void*)src.data(), src.stride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src,
               Matx<_Tp, _rows, _cols>& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        dst = Matx<_Tp, _cols, _rows>(static_cast<const _Tp*>(src.data())).t();
    }
    else
    {
        dst = Matx<_Tp, _rows, _cols>(static_cast<const _Tp*>(src.data()));
    }
}

int main(int argc, char *argv[])
{
    //~ string datapath = "/home/nikhil/git_repo/arpg/libelas/imgs/MH_01_easy/";
    string datapath = "/home/nikhil/git_repo/arpg/libelas/imgs/V1_01_easy/";

    string filepath0 = datapath + "mav0/cam0/sensor.yaml";
    string filepath1 = datapath + "mav0/cam1/sensor.yaml";
    cv::FileStorage fs0(filepath0, cv::FileStorage::READ);
    cv::FileStorage fs1(filepath1, cv::FileStorage::READ);
    Mat distCoeffs;
    string type;

    if (fs0.isOpened() && fs1.isOpened())
        cout << "Config files opened\n";
    else
        cout << "Could not open config files\n";

    FileNode T_BS_0 = fs0["T_BS"]["data"];
    FileNode T_BS_1 = fs1["T_BS"]["data"];
    FileNode INTR_0 = fs0["intrinsics"];
    FileNode INTR_1 = fs1["intrinsics"];
    FileNode DIST_COEFF_0 = fs0["distortion_coefficients"];
    FileNode DIST_COEFF_1 = fs1["distortion_coefficients"];

    Mat cam_mat0 = (Mat_<float>(3, 3) << INTR_0[0], 0.0, INTR_0[2], 0.0, INTR_0[1], INTR_0[4], 0.0, 0.0, 1.0);
    Mat cam_mat1 = (Mat_<float>(3, 3) << INTR_1[0], 0.0, INTR_1[2], 0.0, INTR_1[1], INTR_1[4], 0.0, 0.0, 1.0);
    Mat dist_coeff0 = (Mat_<float>(1, 4) << DIST_COEFF_0[0],  DIST_COEFF_0[1], DIST_COEFF_0[2], DIST_COEFF_0[3]);
    Mat dist_coeff1 = (Mat_<float>(1, 4) << DIST_COEFF_1[0],  DIST_COEFF_1[1], DIST_COEFF_1[2], DIST_COEFF_1[3]);

    Eigen::Matrix4f T_BS_e1;
    T_BS_e1 << T_BS_0[0], T_BS_0[1], T_BS_0[2], T_BS_0[3],
               T_BS_0[4], T_BS_0[5], T_BS_0[6], T_BS_0[7],
               T_BS_0[8], T_BS_0[9], T_BS_0[10], T_BS_0[11],
               T_BS_0[12], T_BS_0[13], T_BS_0[14], T_BS_0[15];

    Eigen::Matrix4f T_BS_e2;
    T_BS_e2 << T_BS_1[0], T_BS_1[1], T_BS_1[2], T_BS_1[3],
               T_BS_1[4], T_BS_1[5], T_BS_1[6], T_BS_1[7],
               T_BS_1[8], T_BS_1[9], T_BS_1[10], T_BS_1[11],
               T_BS_1[12], T_BS_1[13], T_BS_1[14], T_BS_1[15];

    Eigen::Matrix4f T_BS_e = T_BS_e1.inverse() * T_BS_e2;
    Eigen::Matrix3f T_BS_r = T_BS_e.block<3,3>(0,0);
    Eigen::MatrixXf T_BS_t = T_BS_e.block<3,1>(0,3);;

    Mat rot_mat, trans_vec;
    eigen2cv(T_BS_r, rot_mat);
    eigen2cv(T_BS_t, trans_vec);


    vector<string> filenames;

    // MH_01_easy
    //~ filenames.push_back("1403636581413555456");
    //~ filenames.push_back("1403636621913555456");
    //~ filenames.push_back("1403636654413555456");
    //~ filenames.push_back("1403636735913555456");

    // V1_01_easy
    filenames.push_back("1403715287412143104");
    filenames.push_back("1403715303912143104");
    filenames.push_back("1403715312912143104");
    filenames.push_back("1403715404912143104");

    for (int ii=0; ii<filenames.size(); ii++)
    {
        Mat im0 = imread(datapath + "mav0/cam0/data/" + filenames[ii] + ".png", CV_LOAD_IMAGE_UNCHANGED);
        Mat im1 = imread(datapath + "mav0/cam1/data/" + filenames[ii] + ".png", CV_LOAD_IMAGE_UNCHANGED);
    
        cam_mat0.convertTo(cam_mat0, CV_64F);
        dist_coeff0.convertTo(dist_coeff0, CV_64F);
        cam_mat1.convertTo(cam_mat1, CV_64F);
        dist_coeff1.convertTo(dist_coeff1, CV_64F);
        rot_mat.convertTo(rot_mat, CV_64F);
        trans_vec.convertTo(trans_vec, CV_64F);
    
        Size im_size = im1.size();
        Mat new_im0, new_im1, map01, map02, map11, map12, R0, R1, P0, P1, Q;
    
        stereoRectify(cam_mat0, dist_coeff0, cam_mat1, dist_coeff1, im_size,
                      rot_mat, trans_vec, R0, R1, P0, P1, Q);
    
        initUndistortRectifyMap(cam_mat0, dist_coeff0, R1, P1, im_size, im0.type(), map01, map02);
        initUndistortRectifyMap(cam_mat1, dist_coeff1, R0, P0, im_size, im1.type(), map11, map12);
    
        remap(im0, new_im0, map01, map02, INTER_LINEAR);
        remap(im1, new_im1, map11, map12, INTER_LINEAR);
    
        Mat disp, dispn;
    
        StereoBM sbm;
        sbm.state->SADWindowSize = 9;
        sbm.state->numberOfDisparities = 112;
        sbm.state->preFilterSize = 5;
        sbm.state->preFilterCap = 61;
        sbm.state->minDisparity = -39;
        sbm.state->textureThreshold = 507;
        sbm.state->uniquenessRatio = 0;
        sbm.state->speckleWindowSize = 0;
        sbm.state->speckleRange = 8;
        sbm.state->disp12MaxDiff = 1;

        sbm(new_im0, new_im1, disp);
        normalize(disp, dispn, 0, 255, CV_MINMAX, CV_8U);

        /*
        namedWindow( "new_img0", WINDOW_AUTOSIZE);
        imshow( "new_img0", new_im0);
    
        namedWindow( "new_img1", WINDOW_AUTOSIZE);
        imshow( "new_img1", new_im1);
    
        namedWindow( "sbm", WINDOW_AUTOSIZE);
        imshow( "sbm", dispn);
    
        waitKey(0);
        destroyAllWindows();
        */

        std::string index;
        std::ostringstream convert;
        convert << ii;
        index = convert.str();

        string left_file = datapath + "mav0/rectified/" + index + "_l.pgm";
        string right_file = datapath + "mav0/rectified/" + index + "_r.pgm";
    
        cv::imwrite(left_file, new_im0);
        cv::imwrite(right_file, new_im1);
    
        fs0.release();
        fs1.release();
    }
    cout << "Done\n";
    return 0;
}
