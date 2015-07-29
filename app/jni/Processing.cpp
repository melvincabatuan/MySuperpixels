#include "com_cabatuan_mysuperpixels_MainActivity.h"
#include <android/log.h>
#include <android/bitmap.h>
#include <stdlib.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "Superpixels.hpp"

using namespace cv;

#define  LOG_TAG    "MySuperpixels"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

// AVERAGE: 13.861056169668833 fps after 600 frames
#define max(pValue1, pValue2) \
  (pValue1 ^ ((pValue1 ^ pValue2) & -(pValue1 < pValue2)))

#define clamp(pValue, pLowest, pHighest) \
 ((pValue < 0) ? pLowest : (pValue > pHighest) ? pHighest: pValue)

// Get the most significant 8 bits per color and concatenate alpha
#define color(pColorR, pColorG, pColorB) \
           (0xFF000000 | ((pColorB << 6)  & 0x00FF0000) \
                       | ((pColorG >> 2)  & 0x0000FF00) \
                       | ((pColorR >> 10) & 0x000000FF))
                       
                       
/*Global variable*/                       
                       
                       

static void outputGray( const cv::Mat &src, uint32_t* bitmapContent){

   uint32_t R, G, B;

   int nRows = src.rows;
   int nCols = src.cols;


   if (src.isContinuous()) {
        nCols = nCols * nRows;
	 	nRows = 1;  
	 	
   }   

   int32_t i, j, index;
 

   for (j = 0, index = 0; j < nRows; ++j){
   
     const char* row  = src.ptr<char>(j); 
   
   		 for (i = 0; i < nCols; ++i, ++index){
   		 
   		    R = 1192 * 220;   //  220 - Studio swing; 255 - Full swing
   		    G = 1192 * 0;   // 
   		    B = 1192 * 0;   //
   		 
   		    R = clamp(R, 0, 262143);  // 262143 18 1 bit binary
            G = clamp(G, 0, 262143);
            B = clamp(B, 0, 262143);
   		 
   		     bitmapContent[index] = color( R, G, B);
   		                                 
   		 
   		 }   
   
   }


}


// AVERAGE: 15.52 fps after 600 frames (orig max)
// cv::Mat &imageYUV is scaled by 2

// AVERAGE: 10.76 fps after 440 frames (bitwise max)

static void extractYUV(const cv::Mat &srcNV21, cv::Mat &imageYUV){

   size_t j, i;
   size_t nRows = 2 * srcNV21.rows / 3; // number of lines
   size_t nCols = srcNV21.cols;   // number of columns 
   size_t uvRowIndex, uvColIndex;
   
   uchar Ymax;
    
   for (j = 0, uvRowIndex = 0; j < nRows - 1; j+=2, uvRowIndex++) {    
      
      const uchar* current = srcNV21.ptr<const uchar>(j);   // current NV21 row
      const uchar* next    = srcNV21.ptr<const uchar>(j+1); // next NV21 row  
           
      const uchar* uv_row  = srcNV21.ptr<const uchar>(nRows + uvRowIndex); // uv row
      
      cv::Vec4b *yuv_row   = imageYUV.ptr<cv::Vec4b>(uvRowIndex);     
    
      for (i = 0, uvColIndex = 0; i < nCols - 1; i += 2, uvColIndex++) {
      
          // Get max (max pooling)          
             Ymax = cv::saturate_cast<uchar>(max(current[i], current[i+1]));
             Ymax = cv::saturate_cast<uchar>(max(Ymax, next[i]));
             Ymax = cv::saturate_cast<uchar>(max(Ymax, next[i+1]));
                 
          // Assign to a pixel
             yuv_row[uvColIndex] = cv::Vec4b(Ymax, uv_row[i+1], uv_row[i], 255); 
       }
         
    }
}



 


// AVERAGE: 16.46 fps after 600 frames
void extractVU_method2(const cv::Mat &srcNV21, cv::Mat &imageYUV){

    Mat Y, U, V;
    
    size_t height = 2*srcNV21.rows/3;
    
    // Luma
        Y     = srcNV21.rowRange( 0, height);
    
    // Chroma U
    if (U.empty())
    	U.create(cv::Size(srcNV21.cols/2, height/2), CV_8UC1);
    
    // Chroma V
    if (V.empty())
    	V.create(cv::Size(srcNV21.cols/2, height/2), CV_8UC1);   	    
    
    Mat image = srcNV21.rowRange( height, srcNV21.rows);
	size_t nRows = image.rows;   // number of lines
    size_t nCols = image.cols;   // number of columns  

    /// Convert to 1D array if Continuous
    if (image.isContinuous()) {
        nCols = nCols * nRows;
		nRows = 1; // it is now a 
	}   

    for (int j=0; j<nRows; j++) {
    
        /// Pointer to start of the row      
        uchar* data   = reinterpret_cast<uchar*>(image.data);
        uchar* colorV = reinterpret_cast<uchar*>(V.data);
        uchar* colorU = reinterpret_cast<uchar*>(U.data);

		for (int i = 0; i < nCols; i += 2) {
		        // assign each pixel to V and U
                *colorV++ = *data++; //  [0,255]
                *colorU++ = *data++; //  [0,255]   
        }
    }
    
    std::vector<cv::Mat> channels(4);
    
    cv::Mat Yscaled;
    
    cv::resize(Y, Yscaled, cv::Size(srcNV21.cols/2, height/2));    

    channels[0] = Yscaled;
    channels[1] = U;
    channels[2] = V;
    channels[3] = Mat::zeros(cv::Size(srcNV21.cols/2, height/2), CV_8UC1) + 255;

    cv::merge(channels, imageYUV);
}



 
void extractVU_method3(const cv::Mat &srcNV21, cv::Mat &imageYUV){

    Mat Y, U, V;
    
    size_t height = 2*srcNV21.rows/3;
    
    // Luma
        Y     = srcNV21.rowRange( 0, height);
    
    // Chroma U
    if (U.empty())
    	U.create(cv::Size(srcNV21.cols/2, height/2), CV_8UC1);
    
    // Chroma V
    if (V.empty())
    	V.create(cv::Size(srcNV21.cols/2, height/2), CV_8UC1);   	    
    
    Mat image = srcNV21.rowRange( height, srcNV21.rows);
	size_t nRows = image.rows;   // number of lines
    size_t nCols = image.cols;   // number of columns  

    /// Convert to 1D array if Continuous
    if (image.isContinuous()) {
        nCols = nCols * nRows;
		nRows = 1; // it is now a 
	}   

    for (int j=0; j<nRows; j++) {
    
        /// Pointer to start of the row      
        uchar* data   = reinterpret_cast<uchar*>(image.data);
        uchar* colorV = reinterpret_cast<uchar*>(V.data);
        uchar* colorU = reinterpret_cast<uchar*>(U.data);

		for (int i = 0; i < nCols; i += 2) {
		        // assign each pixel to V and U
                *colorV++ = *data++; //  [0,255]
                *colorU++ = *data++; //  [0,255]   
        }
    }
    
    std::vector<cv::Mat> channels(3);
    
    cv::Mat Yscaled;
    
    cv::resize(Y, Yscaled, cv::Size(srcNV21.cols/2, height/2));    

    channels[0] = Yscaled;
    channels[1] = U;
    channels[2] = V;

    cv::merge(channels, imageYUV);
}


/* Global Variables */

cv::Mat lut;

void colorReduction(const cv::Mat& src, cv::Mat& dst, int div = 128) {
          
      if(lut.empty())
           lut.create(1,256,CV_8U);

      uchar* rdata = reinterpret_cast<uchar*>(lut.data);
      
      for (int i = 0; i<256; i++) {
          *rdata++ = i/div*div + div >> 1;
      }
     
      cv::LUT( src, lut, dst);
}









Superpixels s;
Mat tempBGR;

/*
 * Class:     com_cabatuan_mysuperpixels_MainActivity
 * Method:    filter
 * Signature: (Landroid/graphics/Bitmap;[B)V
 */
JNIEXPORT void JNICALL Java_com_cabatuan_mysuperpixels_MainActivity_process
  (JNIEnv *pEnv, jobject clazz, jobject pTarget, jbyteArray pSource, jint compactness){

   AndroidBitmapInfo bitmapInfo;
   uint32_t* bitmapContent; // Links to Bitmap content

   if(AndroidBitmap_getInfo(pEnv, pTarget, &bitmapInfo) < 0) abort();
   if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) abort();
   if(AndroidBitmap_lockPixels(pEnv, pTarget, (void**)&bitmapContent) < 0) abort();

   /// Access source array data... OK
   jbyte* source = (jbyte*)pEnv->GetPrimitiveArrayCritical(pSource, 0);
   if (source == NULL) abort();

   Mat src(bitmapInfo.height + bitmapInfo.height/2, bitmapInfo.width, CV_8UC1, (unsigned char *)source);
   Mat mbgra(bitmapInfo.height, bitmapInfo.width, CV_8UC4, (unsigned char *)bitmapContent);
   
   
/***********************************************************************************************/ 
   
   //outputGray(src.rowRange(0, bitmapInfo.height), bitmapContent);
   
   if (tempBGR.empty())
       tempBGR.create(bitmapInfo.height/2, bitmapInfo.width/2, CV_8UC3);
       
    //extractYUV(src, tempBGRA);  
   
    extractVU_method3(src, tempBGR); // YUV extraction OK
    
    // Reduce color
    colorReduction(tempBGR, tempBGR, 4);
    
    s.generateSuperpixels(tempBGR, compactness);
    
    cv::Mat BGRscaled;    
    cv::resize(tempBGR, BGRscaled, mbgra.size());
    
    //s.displayCenters(BGRscaled, cv::Scalar( 0, 255, 0)); // OK
    
    s.displayContours(BGRscaled);
    
    cv::cvtColor(BGRscaled, mbgra, CV_BGR2BGRA);
    
/***********************************************************************************************/
    
 
     // Sanity check: okay!
    //cvtColor(src.rowRange(0, bitmapInfo.height), mbgra, CV_GRAY2BGRA);
     

/************************************************************************************************/ 
   
   /// Release Java byte buffer and unlock backing bitmap
   pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();
}
