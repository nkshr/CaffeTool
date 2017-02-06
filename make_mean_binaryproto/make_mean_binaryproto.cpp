#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/opencv.hpp>

using namespace caffe;
using namespace cv;
using namespace std;

int main(int argc, char ** argv){
  if(argc != 3){
    cerr << "Invalid arguments." << endl;
    cerr << "Usage : " << endl;
    cerr << "./make_mean_binaryproto <imgs_list> <binaryproto>" << endl;
    return 1;
  }

  char * imgs_list_name = argv[1];
  char * bp_name = argv[2];

  ifstream imgs_list;
  if(!imgs_list.good()){
    cerr << "Couldn't open " << imgs_list_name << endl;
    return 1;
  }

  Mat sum_img = Mat::zeros(256, 256, CV_32F);
  int num_imgs = 0;
  while(true){
    string img_name;
    imgs_list >> img_name;
    Mat img = imread(img_name);
    if(img.empty()){
      cerr << "CouldN't load " << img_name << endl;
      return 1;
    }
    MatIterator_<uchar> it_img = img.begin<uchar>();
    MatIterator_<uchar> end = img.end<uchar>();
    MatIterator_<float> it_sum_img = sum_img.begin<float>();
    for(; it_img != end; ++it_img){
      (*it_sum_img) += (float)(*it_img);
    }
    num_imgs++;
    if(num_imgs % 1000){
      cout << num_imgs << " images has been added." << endl;
    }
  }
 
  cout << "total number of images is " << num_imgs << endl;

  MatIterator_<float> it_sum_img = sum_img.begin<float>();
  MatIterator_<float> end = sum_img.end<float>();
  const float inum_imgs = 1.f / (float)num_imgs;
  for(; it_sum_img != end; ++it_sum_img){
    (*it_sum_img) *= inum_imgs;
  }

  BlobProto img_blob;
  img_blob.set_num(1);
  img_blob.set_channels(sum_img.channels());
  img_blob.set_height(sum_img.rows);
  img_blob.set_width(sum_img.cols);

  for(int i = 0; i < sum_img.rows * sum_img.cols * sum_img.channels(); ++i){
    img_blob.add_data(0.);
  }
  
  const float * psum_img = sum_img.ptr<float>(0);
  for(int c = 0; c < sum_img.channels(); ++c){
    for(int i = 0; i < sum_img.total(); ++i){
      img_blob.set_data(c*sum_img.total() + i, psum_img[3*i+ c]);
    }
  }
  WriteProtoToBinaryFile(img_blob, bp_name);
  return 0;    
}
