#pragma warning(disable: 4819)

#include <seeta/FaceEngine.h>

#include <seeta/Struct_cv.h>
#include <seeta/Struct.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <array>
#include <map>
#include <iostream>
#include <ctime>
#include <iomanip>

// #ifdef __APPLE__
    #include <sys/uio.h>
// #else
//     #include <sys/io.h>
// #endif
#include <fstream>

#include <seeta/QualityAssessor.h>
#include <ClarityQuality.h>

using namespace std;

int main()
{
    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    int id = 0;

    seeta::QualityAssessor QA;

    // recognization threshold
    float threshold = 0.7f;


    std::string filename = "/Users/jinyufeng/Projects/SeetaFace2/example/search/1.jpg";
    cv::Mat frame = cv::imread(filename);
    seeta::cv::ImageData image = frame;
    // auto score = QA.evaluate(image, face.pos, points.data());

    SeetaRect face;
    face.x = 0;
    face.y = 0;
    face.width = frame.cols;
    face.height = frame.rows;
    float score = evaluate_clarity(image, face);

    std::cout<<"score========="<<score<<std::endl;


    // std::string txtfile = "/Users/jinyufeng/Downloads/20230801_v2_crop_faces.txt";
    // string savefolder = "/Users/jinyufeng/Downloads/20230801_v2_crop_faces_seetafaceQua";

    std::string txtfile = "/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2.txt";
    string savefolder = "/Users/jinyufeng/Downloads/20230801_v2_align_faces_v2_seetafaceQua";

    std::vector<std::string> imgfile;
    string need_extension = ".txt";
    size_t pos = txtfile.find(need_extension);
    string folder = txtfile.substr(0, pos);
    std::cout<<"folder========="<<folder<<std::endl;
    ifstream infile;
    infile.open(txtfile.data());
    string linestr;
    int idx = 0;
    float total_time = 0.0;
    clock_t begin, end;
    while(getline(infile, linestr))
    {
        idx+=1;
        if(idx%1000==0){
            cout<<"processed "<<idx<<" images"<<endl;
        }
        filename = folder + '/' + linestr;
        // cout<<filename<<endl;
        cv::Mat frame = cv::imread(filename);
        seeta::cv::ImageData image = frame;
        // SeetaRect face;
        face.x = 0;
        face.y = 0;
        face.width = frame.cols;
        face.height = frame.rows;
        begin = clock();
        score = evaluate_clarity(image, face);
        end = clock();
        // cout << "cost time: " << double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        total_time += double(end - begin) / CLOCKS_PER_SEC * 1000;

        // std::cout<<"score========="<<score<<std::endl;
        cout<<setiosflags(ios::fixed)<<setprecision(3);
        string blur_label = to_string(score);
        // std::cout<<"blur_label========="<<blur_label<<std::endl;
        string save_path = savefolder + '/' + blur_label + "_" + linestr;
        // std::cout<<"save_path========="<<save_path<<std::endl;
        cv::imwrite(save_path, frame);
        
    }
    infile.close();
    cout<<"total time: "<<total_time<<endl;

}
