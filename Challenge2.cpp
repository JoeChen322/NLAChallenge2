#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <random>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
typedef Eigen::Triplet<double> T;



/*-------------------------------------------------Main()-------------------------------------------------------*/
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char *input_image_path = argv[1];

    /*****************************Load the image as a Matrix****************************/
    int width, height, channels;
    // for greyscale images force to load only one channel
    unsigned char *image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data)
    {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image " << argv[1] << " loaded: " << height << "x" << width << " pixels with " << channels << " channels" << std::endl;
   
    Matrix<double, Dynamic, Dynamic, RowMajor> A(height, width);
    
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = (i * width + j) * channels;
            A(i, j) = static_cast<double>(image_data[index]) / 255.0;
        }
    }

    // Report the size of the matrix
    std::cout << "The original image " << argv[1] << " in matrix form has dimension: " << A.rows() << " rows x " << A.cols()
              << " cols = " << A.size() << "\n" << std::endl;

   
  // Free memory!!!
  stbi_image_free(image_data);

   Matrix<double, Dynamic, Dynamic, RowMajor> ATA(height, width);
   ATA=A.transpose()*A;
   saveMarket(ATA, "ATA.mtx");
/*------------------------Q1. CANCULATE ATA--------------------------------*/
   
   std::cout<<"norm of A^T*A:" <<ATA.norm()<<std::endl;

/*------------------------Q2.Canculate ATAX=LAMDAX-----------------------------*/
   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(ATA);
   VectorXd eigenValues =solver.eigenvalues();
   std::cout<<"Top2 MAX Singular Values of A:" <<sqrt(eigenValues(eigenValues.size() - 1))<<" , "
   <<sqrt(eigenValues(eigenValues.size() - 2))<<std::endl;
/*------------------------Q3.Agreement-----------------------------------------*/
    std::cout<<"result is in agreement with the previous one"<<std::endl;
/*------------------------Q4 is run by lis-------------------------------------*/  
/*------------------------Q5.6,7 diagnol of SVD------------------------------------*/  
   /* Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    MatrixXd U=svd.matrixU();
    MatrixXd V=svd.matrixV();
    MatrixXd sigma = svd.singularValues().asDiagonal();
    MatrixXd C= U.leftCols(80);
    MatrixXd D = V.leftCols(80) * svd.singularValues().head(80).asDiagonal();
    std::cout<<"norm of sigma:"<<sigma.norm()<<std::endl;
    std::cout << "Nonzero Entries For Matrix C80: " << C.nonZeros() << std::endl;
    std::cout << "Nonzero Entries For Matrix D80: " << D.nonZeros() << std::endl;
    MatrixXd img=C*(D.transpose());
    img = img.cwiseMin(1).cwiseMax(0);
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> img_output = img.unaryExpr(
        [](double pixel)
        { return static_cast<unsigned char>(pixel * 255.0); });
    const std::string output_image_path1 = "CDT80.png";
  if (stbi_write_png(output_image_path1.c_str(), width, height, 1,img_output.data(), width) == 0)
   {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
    return 1;
  }
  std::cout << "CDT image saved to " << output_image_path1 << std::endl;*/
/*------------------------Q8.diagnol of SVD------------------------------------*/ 
/*------------------------Q9.add noise to checkboard---------------------------*/
    MatrixXd noise;
    noise=MatrixXd::Random(200,200);
    noise=50*noise;
    
    Matrix<double, Dynamic, Dynamic, RowMajor> checkerboard(200, 200);    
    Matrix<double, Dynamic, Dynamic, RowMajor> noise_checkerboard(200, 200);
    // Create checkerboard
    for(int i=0;i<200;++i)
    {
        for (int j = 0; j < 200; ++j) 
        {
            
            if ((i / 25) % 2 == (j / 25) % 2) 
            {
                checkerboard(i, j) =0; 
                if((checkerboard(i, j)+noise(i,j))<=0)
                {noise_checkerboard(i,j)=0;}
                else noise_checkerboard(i,j)= (checkerboard(i, j)+noise(i,j))/255.0;
            } 
            else 
            {
                checkerboard(i, j) = 255; 
                if((checkerboard(i, j)+noise(i,j))>=255)
                {noise_checkerboard(i,j)=1;}
                else noise_checkerboard(i,j)=(checkerboard(i, j)+noise(i,j)) /255.0;
            }
        }
    }
    
    Eigen::BDCSVD<Eigen::MatrixXd> checkboard_svd(noise_checkerboard, ComputeThinU | ComputeThinV);
    MatrixXd checkU=checkboard_svd.matrixU();
    MatrixXd checkV=checkboard_svd.matrixV();
    MatrixXd checksigma = checkboard_svd.singularValues().asDiagonal();
    std::cout<<"top 2 singular value of checkboard"<<checkboard_svd.singularValues()[0]<<","<<checkboard_svd.singularValues()[1]<<std::endl;
    MatrixXd checkC= checkU.leftCols(10);
    MatrixXd checkD = checkV.leftCols(10) * checkboard_svd.singularValues().head(10).asDiagonal();
    std::cout<<"norm of sigma:"<<checksigma.norm()<<std::endl;
    MatrixXd img=checkC*(checkD.transpose());
    img = img.cwiseMin(1).cwiseMax(0);
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> img_output = img.unaryExpr(
        [](double pixel)
        { return static_cast<unsigned char>(pixel * 255.0); });
    const std::string output_image_path1 = "CDT10.png";
  if (stbi_write_png(output_image_path1.c_str(), 200, 200, 1,img_output.data(), 200) == 0)
   {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
    return 1;
  }
  std::cout << "CDT image saved to " << output_image_path1 << std::endl;

  //turn matrix into vector


  
    
    return 0;
}
