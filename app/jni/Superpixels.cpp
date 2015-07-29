#include "Superpixels.hpp"


/*
 * Default Constructor
 */
Superpixels::Superpixels(){
  // Do Nothing...
}



/*
 * Destructor. Clear any present data.
 */
Superpixels::~Superpixels(){

    centers.clear(); // vector
                     // Mat data structure will handle grabage collection itself 
}



void Superpixels::initialize(const cv::Mat &image) { 

    this->m = DEFAULT_m;
    
    /* Initialize the clusters. */
    this->nx = DEFAULT_nx; // number of clusters in x
    this->ny = DEFAULT_ny; // number of clusters in y
    
    /*  Horizontal and Vertical steps */
    float dx = image.cols / float(nx); // x steps
    float dy = image.rows / float(ny); // y steps
    
    this->step = (dx + dy + 1)/2; // default window size S; 
                                   // Note: search space will be 2S x 2S 
              
             
    /* Initialize the centers. */
	for (int i = 0; i < ny; i++) {    // i cell rows
		for (int j = 0; j < nx; j++) { // j cell cols
		      this->centers.push_back( Point(j*dx+dx/2, i*dy+dy/2) );
		}
	}
	 
	
	/* Initialize the cluster labels. */
    labels = Mat::zeros(image.size(), CV_8UC1); 
    
    
    /* Initialize the distance matrix. */
	  distances  = UINT_MAX * Mat::ones(image.size(), CV_16UC1); // UINT_MAX =  65535  
}



// OK: Tested
void Superpixels::displayCenters(cv::Mat &image, cv::Scalar color) {
    
    /// Scale is 2
    for (size_t i = 0; i < centers.size(); i++) {
        cv::circle(image,                              // destination image
        cv::Point(centers[i].x * 2, centers[i].y * 2), // center coordinate
        2,                                             // radius
        color, 										   // color
        2);											   // thickness
    }  
    
    cv::putText(image, 									// destination image
	"Superpixel Centers", 								// text
	cv::Point( image.cols/4, image.rows/2 ), 			// text position
	cv::FONT_HERSHEY_PLAIN, 							// font type
	1.0, 												// font scale
	cv::Scalar(0,0,0,255), 								// text color  
	1); 												// text thickness 
}



/// YUV image input
void Superpixels::generateSuperpixels(cv::Mat &image){

   /// Clear previous centers
   centers.clear();

   /// Initialize clusters, centers, and labels
   initialize(image);
   
   // Mat image_roi, labels_roi, distances_roi;   
   Point2i pcenter;
   
   /// nx * ny : total clusters; same as number of centers; e.x. 15*15 = 225 clusters
   uchar n = nx * ny;
   
   // 10 iterations as suggested by the SLIC authors
  // for (size_t k = 0; k < 5; k++){
   
   
   /// Iterate through all clusters
   for (size_t c = 0; c < n; c++){ // for all cluster
   
         // Current center
         pcenter = centers[c]; 
         
         int xmin = max<int>(pcenter.x - step, 0);
         int ymin = max<int>(pcenter.y - step, 0);
         int xmax = min<int>(pcenter.x + step, image.cols - 1);
         int ymax = min<int>(pcenter.y + step, image.rows - 1);
         
         // Search in a roi window around the center
                      
         /// Compute distances from center for each pixels in roi                
         /// If any pixel distance from the cluster center is less than its
         /// previous value update its distance and label  
         
         for(int j = ymin; j < ymax; j++) { // iterate roi rows
    
        	 const cv::Vec4b *img_row = image.ptr<cv::Vec4b>(j);
        	 const cv::Vec4b current_center = image.ptr<cv::Vec4b>(pcenter.y)[pcenter.x];
        	 uchar *lbl_row   = labels.ptr<uchar>(j);
        	 uint16_t *dst_row  = distances.ptr<uint16_t>(j);
        
        	 for(int i = xmin; i < xmax; i++){ // iterate roi cols
        	
        		// Compute distance from center color   
        		uint16_t current_distance = computeManhattanDistance(img_row[i], current_center, pcenter.x - i, pcenter.y - j);     
     			
		   		if (current_distance < dst_row[i]) {
		   		
		   			// Update distances
					dst_row[i] = current_distance;
		   	
					// Update labels
					lbl_row[i] = saturate_cast<uchar>(c);    // Clusters, {c0, c1, c2, ..., cn-1}	
					                   // Normalize labels later into [0,255]
					                  									
				}  // END if 
								
        	} // END inner for       	
        	      
         } // END cluster c
                               
   } // END for all cluster
  
   
     /// Update cluster centers, c, by finding the center of mass 
       updateCenters();
         
      //updateCentersAndMorphology(); // with clean-up: a bit slow
            
   
   // } /// END 10 iterations
   
} 






/// YUV image input
void Superpixels::generateSuperpixels(cv::Mat &image, int compactness, int texture){

   /// Clear previous centers
   centers.clear();
   
   /// Set compactness
   this->m = compactness;
   
   /// Set texture shift
   this->txt = texture;

   /// Initialize clusters, centers, and labels
   initialize(image);
   
   // Mat image_roi, labels_roi, distances_roi;   
   Point2i pcenter;
   
   /// nx * ny : total clusters; same as number of centers; e.x. 15*15 = 225 clusters
   uchar n = nx * ny;
   
   // 10 iterations as suggested by the SLIC authors
    for (size_t k = 0; k < 5; k++){
   
   
   /// Iterate through all clusters
   for (size_t c = 0; c < n; c++){ // for all cluster
   
         // Current center
         pcenter = centers[c]; 
         
         int xmin = max<int>(pcenter.x - step, 0);
         int ymin = max<int>(pcenter.y - step, 0);
         int xmax = min<int>(pcenter.x + step, image.cols - 1);
         int ymax = min<int>(pcenter.y + step, image.rows - 1);
         
         // Search in a roi window around the center                      
         /// Compute distances from center for each pixels in roi                
         /// If any pixel distance from the cluster center is less than its
         /// previous value update its distance and label  
         
         for(int j = ymin; j < ymax; j++) { // iterate roi rows
    
        	 const cv::Vec4b *img_row = image.ptr<cv::Vec4b>(j);
        	 const cv::Vec4b current_center = image.ptr<cv::Vec4b>(pcenter.y)[pcenter.x];
        	 uchar *lbl_row   = labels.ptr<uchar>(j);
        	 uint16_t *dst_row  = distances.ptr<uint16_t>(j);
        
        	 for(int i = xmin; i < xmax; i++){ // iterate roi cols
        	
        		// Compute distance from center color
        		   
        		 uint16_t current_distance = computeManhattanDistance(img_row[i], current_center, pcenter.x - i, pcenter.y - j); 
        		
        		//uint16_t current_distance = computeSquaredDistance(img_row[i], current_center, pcenter.x - i, pcenter.y - j);   

        		
        		// dx = p1.x - i cols
         		// dy = p1.y - j rows     
     			
		   		if (current_distance < dst_row[i]) {
		   		
		   			// Update distances
					dst_row[i] = current_distance;
		   	
					// Update labels
					lbl_row[i] = saturate_cast<uchar>(c);    // Clusters, {c0, c1, c2, ..., cn-1}	
					                   // Normalize labels later into [0,255]
					                  									
				}  // END if 
								
        	} // END inner for       	
        	      
         } // END cluster c
                               
   } // END for all cluster
  
   
     /// Update cluster centers, c, by finding the center of mass 
        updateCenters(); // reached 15 fps
         
     //updateCentersAndMorphology(); // wiht clean-up: a bit slow
            
   
   } /// END 10 iterations
   
} 



 
 



void Superpixels::updateCenters(){

        Mat cluster_mask;
 
    /// Iterate through all clusters
    /// nx * ny : total clusters; same as number of centers; 
        uchar n = nx * ny;   
        
        for (size_t c = 0; c < n; c++){ // for all cluster     
        
                   
        	int xmin = max<int>(centers[c].x - step, 0);
         	int ymin = max<int>(centers[c].y - step, 0);
         	int xmax = min<int>(centers[c].x + step, labels.cols - 1);
         	int ymax = min<int>(centers[c].y + step, labels.rows - 1);
         	
       	 
        
    /// Compute label mask
            cluster_mask = (labels(Range(ymin, ymax), Range(xmin, xmax))) == saturate_cast<uchar>(c);    
    
    /// Get centroid using moments
            cv::Moments m = moments(cluster_mask, true);             

    /// Update new center to centers   
            // The cause of the error was the offset!  xmin + ____ ,  ymin + _____
            
            cv::Point newCenter( xmin + m.m10/m.m00,  ymin + m.m01/m.m00); 
            
           
            centers[c] = newCenter; 
      
        } // ENDFOR all clusters
 
}

 
 
 
 
 /*
 
 void Superpixels::mergeCenters(int partition){

        Mat cluster_mask;
 
    /// Iterate through all clusters
    /// nx * ny : total clusters; same as number of centers; 
        uchar n = (nx/partition) * (ny/partition);   
        
        for (size_t c = 0; c < n; c++){ // for all cluster     
        
            // std::cout << "old center = ( " << centers[c].x << ", " << centers[c].y <<" )" << std::endl;
                    
        	int xmin = max<int>(centers[c].x - partition * step, 0);
         	int ymin = max<int>(centers[c].y - partition * step, 0);
         	int xmax = min<int>(centers[c].x + partition * step, labels.cols - 1);
         	int ymax = min<int>(centers[c].y + partition * step, labels.rows - 1);
         	
         	// std::cout << "xmin = " << xmin <<", ymin = " <<ymin<< std::endl;         	 
        
    /// Compute label mask
            cluster_mask = (labels(Range(ymin, ymax), Range(xmin, xmax))) == saturate_cast<uchar>(c);    
    
    /// Get centroid using moments
            cv::Moments m = moments(cluster_mask, true);             

    /// Update new center to centers   
            // The cause of the error was the offset!  xmin + ____ ,  ymin + _____
            
            cv::Point newCenter( xmin + m.m10/m.m00,  ymin + m.m01/m.m00); 
            
            // std::cout << "new center = ( " << newCenter.x << ", " << newCenter.y <<" )" << std::endl;
            
            centers[c] = newCenter; 
      
        } // ENDFOR all clusters
 
}

 */
 
 
 
 
 
 
 
void Superpixels::updateCentersAndMorphology(){

        Mat cluster_mask;
        Mat labels_roi;
        
         int morph_size = 4;    
         Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) ); 
        
        //Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
        //Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        //Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        //Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
 
    /// Iterate through all clusters
    /// nx * ny : total clusters; same as number of centers; 
        uchar n = nx * ny;   
        
        for (size_t c = 0; c < n; c++){ // for all cluster     
        
            // std::cout << "old center = ( " << centers[c].x << ", " << centers[c].y <<" )" << std::endl;
                    
        	int xmin = max<int>(centers[c].x - step, 0);
         	int ymin = max<int>(centers[c].y - step, 0);
         	int xmax = min<int>(centers[c].x + step, labels.cols - 1);
         	int ymax = min<int>(centers[c].y + step, labels.rows - 1);
         	
         	// std::cout << "xmin = " << xmin <<", ymin = " <<ymin<< std::endl;  
         	
         	labels_roi = labels(Range(ymin, ymax), Range(xmin, xmax)); 
        
    /// Compute label mask
            cluster_mask = labels_roi == saturate_cast<uchar>(c);  
            
    /// Apply morphology closing; dilate + erode  
           // dilate(cluster_mask, cluster_mask, kernel);
           // erode(cluster_mask, cluster_mask, kernel);
           morphologyEx( cluster_mask,cluster_mask, MORPH_CLOSE, kernel );
            
    /// Update labels in the cluster after clean-up
            Mat lbl(labels_roi.size(), labels_roi.type(), Scalar::all(c));     
            lbl.copyTo(labels_roi, cluster_mask);
            
    /// Get centroid using moments
            cv::Moments m = moments(cluster_mask, true);             

    /// Update new center to centers   
            // The cause of the error was the offset!  xmin + ____ ,  ymin + _____
            
            cv::Point newCenter( xmin + m.m10/m.m00,  ymin + m.m01/m.m00);            
            // std::cout << "new center = ( " << newCenter.x << ", " << newCenter.y <<" )" << std::endl;            
            centers[c] = newCenter; 
           
      
        } // ENDFOR all clusters

}




 

uint16_t Superpixels::computeManhattanDistance(const cv::Vec4b& color, const cv::Vec4b& center, int dx, int dy){

   
    uint16_t d_yuv = abs(color[0]-center[0]) + abs(color[1]-center[1]) + abs(color[2]-center[2]); 
    uint16_t d_xy   = abs(dx) + abs(dy);

    return (d_yuv  + (m/step) * (d_xy));  
}





uint16_t Superpixels::computeSquaredDistance(const cv::Vec4b& color, const cv::Vec4b& center, int dx, int dy){

   
    uint16_t d_yuv = (color[0]-center[0])*(color[0]-center[0]) + (color[1]-center[1])*(color[1]-center[1]) + (color[2]-center[2])*(color[2]-center[2]);
    uint16_t d_lbp  = (color[3]-center[3])*(color[3]-center[3]);
    uint16_t d_xy = dx*dx + dy*dy;
    
    
    return (d_yuv  + (m/step) * (d_xy)  + (d_lbp >> 1));  
}





 


void Superpixels::displayContours(cv::Mat &image){  

    cv::Mat label_scaled;
    
    cv::resize(labels, label_scaled, image.size());

    for(  int j = 1; j < label_scaled.rows - 1; j++ ) {
    
        const uchar* previous = label_scaled.ptr<const uchar>(j-1); // previous row
        const uchar* current  = label_scaled.ptr<const uchar>(j);   // current row 
        const uchar* next     = label_scaled.ptr<const uchar>(j+1); // next row
        
        cv::Vec4b* output = image.ptr<cv::Vec4b>(j);    // output row       
        
        
        for(int i = 1; i < label_scaled.cols - 1; i++) {
            
            const uchar center = label_scaled.ptr<const uchar>(j)[i];
            
            uchar code = 0;

            code |= ((previous[i-1]) != center) << 7;
            code |= ((previous[i])   != center) << 6;
            code |= ((previous[i+1]) != center) << 5;
            code |= ((current[i+1])  != center) << 4;
            code |= ((next[i+1])     != center) << 3;
            code |= ((next[i])       != center) << 2;
            code |= ((next[i-1])     != center) << 1;
            code |= ((current[i-1])  != center) << 0;
            
            // img.at<cv::Vec4b>(row,col)[channel]. reference
            if (bitCount(code) >= 2){ // Change colour to White
                output[i][0] = 255;  //  R
                output[i][1] = 0;  //    G
                output[i][2] = 0;  //    B
                output[i][3] = 255;  //  A
            }
        }
    }
}

 
 
std::vector<Point> Superpixels::getCenters() const{
    return centers;
} 



Mat Superpixels::getLabels() const{
    Mat temp;
    (this->labels).convertTo(temp, CV_8UC1);
    return temp;
}
  
 
  
int Superpixels::bitCount(unsigned char n)
{
  uint i, count = 0;
 
  for (i = 1; i <= sizeof(char) && n; i++)
  {
    count += lookup_table[n & 255];
    n >>= 8;
  }
  return count;
}
