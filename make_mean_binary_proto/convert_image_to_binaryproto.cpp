#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

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
    cerr << "./convert_image_to_binaryproto <img> <binaryproto>" << endl;
    return 1;
  }

  Mat img = imread(argv[1]);
  
  BlobProto img_blob;
  img_blob.set_num(1);
  img_blob.set_channels(img.channels());
  img_blob.set_height(img.rows);
  img_blob.set_width(img.cols);

  for(int i = 0; i < img.rows * img.cols * img.channels(); ++i){
    img_blob.add_data(0.);
  }
  
  const uchar * pimg = img.ptr<uchar>(0);
  for(int c = 0; c < img.channels(); ++c){
    for(int i = 0; i < img.total(); ++i){
      img_blob.set_data(c*img.total() + i, static_cast<float>(pimg[3*i+ c]));
    }
  }
  WriteProtoToBinaryFile(img_blob, argv[2]);
  return 0;    
}
