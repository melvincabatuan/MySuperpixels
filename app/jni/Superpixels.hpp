#include "opencv2/imgproc/imgproc.hpp" 
#include <float.h>  // for FLT_MAX
#include <iostream>
//#include <math.h>


#ifndef MYSUPERPIXELS_H
#define MYSUPERPIXELS_H



/* Default Values*/
  
#define DEFAULT_m 0
//#define DEFAULT_nx 15
//#define DEFAULT_ny 15
#define DEFAULT_nx 8
#define DEFAULT_ny 8

#define DEFAULT_txt 0

/* Very Good Manhattan distance
#define DEFAULT_m 40 
#define DEFAULT_nx 15
#define DEFAULT_ny 15
*/

using namespace cv;

/* Bitcount Look-up */
static const unsigned char lookup_table[256] =
{
    # define B2(n) n,     n+1,     n+1,     n+2
    # define B4(n) B2(n), B2(n+1), B2(n+1), B2(n+2)
    # define B6(n) B4(n), B4(n+1), B4(n+1), B4(n+2)
      B6(0), B6(1), B6(1), B6(2)
};




class Superpixels{

protected:

    Mat labels;   // Labels matrix.               CV_8UC1  // limit the labels to 256 clusters
    Mat distances;// Current distances.           CV_16UC1 // 16 bit signed 
    
    std::vector<Point> centers; // Superpixel centers initially grid centers    
    void updateCenters();       // Update cluster centers to the centroid of labels    
    void updateCentersAndMorphology(); // Update cluster centers to the centroid of labels with clean-up // method is slow...
    
    //void mergeCenters(int partition);  // Ex. partition of 2 means the new step is 2S by 2S for 
                                     // superpixel merging
    
    //float computeDistance(const cv::Vec3b& color, const cv::Vec3b& center, int dx, int dy);
                  // 5-D distance between pixels in YUV space
                  // Euclidean distances          
                  
    uint16_t computeManhattanDistance(const cv::Vec4b& color, const cv::Vec4b& center, int dx, int dy);          // Tested OK results // To be compared
    uint16_t computeSquaredDistance(const cv::Vec4b& color, const cv::Vec4b& center, int dx, int dy);              // Tested OK results // To be compared
    
    int m;        // Compactness parameter or // To be made trackbar
                  // Weighting factor between colour and spatial differences. 
                  // Values from about 5 to 40 are useful.  
                  // Use a large value to enforce superpixels with more regular and
                  // smoother shapes. Try a value of 10 to start with.
                  
                  // m is introduced in Ds allowing us to control the compactness of superpixel.
				  // The greater the value of m, the more spatial proximity is emphasized and 
				  // the more compact the cluster.  
				  // Authors of the algorithm have chosen m = 10.
				  
     int txt;     // texture shift /2 /4 and so on
    
    float step;  //  Window/cell size; Search size is 2S x 2S
    
    uchar nx, ny; //  Window/cell cols and rows     
          
    void initialize(const cv::Mat &image);   //  initialize data  
       
    std::vector<Point> getCenters() const;
    
    int bitCount(unsigned char n);


public:

	Superpixels(); // default constructor 
	~Superpixels();
	
	/*  Display centers for milestone 1  */
	void displayCenters(cv::Mat &image, cv::Scalar color);
	
	/* Return labels normalized [0, 255] */
	cv::Mat getLabels() const;  
	
	/* Generate an over-segmentation for an image. */
    void generateSuperpixels(cv::Mat &image);
    
    /* Generate an over-segmentation for an image with compactness. */
    void generateSuperpixels(cv::Mat &image, int compactness);
    
    /* Generate an over-segmentation for an image with compactness and texture. */
    void generateSuperpixels(cv::Mat &image, int compactness, int texture);
    
    /* Enforce connectivity for an image. */
    void createConnectivity(const cv::Mat &image);
    
    /* Display output contours  */
    void displayContours(cv::Mat &image);
};

#endif
