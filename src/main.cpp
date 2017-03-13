#include <pcl/stereo/disparity_map_converter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
//~ #include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
//~ #include <opencv2/imgcodecs.hpp>
//~ #include <opencv2/highgui.hpp>
//~ #include <opencv2/calib3d.hpp>

#include <calibu/Calibu.h>
#include <HAL/Messages/Image.h>

using namespace pcl;
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    /*
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new 
    pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::RGB>::Ptr left_image (new 
    pcl::PointCloud<pcl::RGB>);
    // Fill left image cloud.

    pcl::DisparityMapConverter<pcl::PointXYZI> dmc;
    dmc.setBaseline (0.8387445f);
    dmc.setFocalLength (368.534700f);
    dmc.setImageCenterX (318.112200f);
    dmc.setImageCenterY (224.334900f);
    dmc.setDisparityThresholdMin(15.0f);

    // Left view of the scene.
    dmc.setImage (left_image);
    // Disparity map of the scene.

    dmc.loadDisparityMap ("/home/nikhil/git_repo/arpg/pcl/disp.pgm", 640, 480);
    dmc.compute(*cloud);
    

    // Loading disparity image
    Mat disparity = imread("/home/nikhil/git_repo/arpg/pcl/disp.pgm", IMREAD_GRAYSCALE);
    if (disparity.empty())
    {
      cerr << "ERROR: Could not read disp.pgm" << std::endl;
      return 1;
    }
    */


    std::shared_ptr<calibu::Rig<double>> cam_xml = calibu::ReadXmlRig("/home/nikhil/git_repo/arpg/pcl/cameras.xml");
    std::shared_ptr<calibu::Rig<double>> rig = calibu::ToCoordinateConvention(cam_xml, calibu::RdfVision);

    float focal_length = (rig->cameras_[0]->GetParams()[0] + rig->cameras_[0]->GetParams()[1])/2.0;
    float baseline = rig->cameras_[1]->Pose().matrix()(0,3);

    Mat img;
    img = imread("/home/nikhil/git_repo/arpg/pcl/disp.pgm");
    //~ img = imread("/home/nikhil/git_repo/arpg/pcl/Playtable-perfect/disp0-n.pgm");
    if(!img.data )
    {
      cout <<  "Could not open or find the image" << endl ;
      return -1;
    }

    const unsigned w = img.size().width;//rig->cameras_[0]->Width();
    const unsigned h = img.size().height;//rig->cameras_[0]->Height();
    cout << "w " << w << " h " << h << endl;

    //~ namedWindow( "Display window", WINDOW_AUTOSIZE );
    //~ imshow( "Display window", image );
    //~ waitKey(0);


    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width=w*h;
    cloud.height=0;
    cloud.is_dense=false;
    cloud.points.resize(cloud.width*(cloud.height+1));

    const unsigned short* d_img = (unsigned short*) img.data;
    Eigen::Vector2d pixel;
    Eigen::Vector3d point;

    float fb = focal_length * baseline;
    float u_centre = (float)rig->cameras_[0]->GetParams()[2]; // h/2.0;
    float v_centre = (float)rig->cameras_[0]->GetParams()[3]; // w/2.0;
    float doffs = (float)rig->cameras_[1]->GetParams()[2]-(float)rig->cameras_[0]->GetParams()[2];
    cout << "f " << focal_length << " | b " << baseline << " | fb " << fb << " | u,v " << u_centre << "," << v_centre << " | doffs " << doffs << endl;

    int zc = 0, nzc = 0;;
    for (unsigned i=0; i<h*w; i++) {

	float u = (float) (i/w);
	float v = (float) (i%w);

        if (d_img[i] == 0) {
            cloud.points[i+cloud.height*h*w].x =
            cloud.points[i+cloud.height*h*w].y =
            cloud.points[i+cloud.height*h*w].z =
            std::numeric_limits<float>::quiet_NaN();
            zc++;
	    continue;
        }


	float depth = fb/((float)img.at<Vec3b>(u,v)[0] + doffs);
	float X = (u-u_centre) * depth / focal_length;
	float Y = (v-v_centre) * depth / focal_length;

        cloud.points[i].x = X;
        cloud.points[i].y = Y;
        cloud.points[i].z = depth;
	nzc++;
    }
    cout << "nzc " << nzc << " | zc " << zc << " | + " << nzc+zc  << " " << endl;
    pcl::io::savePCDFileBinary("pointclouds.pcd", cloud);

    return 0;
}
