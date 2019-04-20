#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

// Sample fingerprint image that is comapred with all other images
string CONTROL_IMG = "101_4.tif"; 

// Threshold for fingerprint comparisions. EX: result <= THRESHOLD is a match
int THRESHOLD = 40;               

/*
 * Helper for thinning() on every iteration
 *
 * @param image : Image loaded of fingerprint
 * @param iter : Iteration of thinning operation
 */
void thinningIteration(Mat& image, int iter)
{
	Mat marker = Mat::zeros(image.size(), CV_8UC1);

	for (int i = 1; i < image.rows - 1; i++) {
		for (int j = 1; j < image.cols - 1; j++) {
			uchar p2 = image.at<uchar>(i - 1, j);
			uchar p3 = image.at<uchar>(i - 1, j + 1);
			uchar p4 = image.at<uchar>(i, j + 1);
			uchar p5 = image.at<uchar>(i + 1, j + 1);
			uchar p6 = image.at<uchar>(i + 1, j);
			uchar p7 = image.at<uchar>(i + 1, j - 1);
			uchar p8 = image.at<uchar>(i, j - 1);
			uchar p9 = image.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
					(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
					(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
					(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
				marker.at<uchar>(i, j) = 1;
			}
		}
	}

	image &= ~marker;
}

/*
 * Thins fingerprint image for more precise comparison
 *
 * @param image : Image loaded of fingerprint
 */
void thinning(Mat &image)
{
	image /= 255;

	Mat prev = Mat::zeros(image.size(), CV_8UC1);
	Mat diff;

	do {
		thinningIteration(image, 0);
		thinningIteration(image, 1);
		absdiff(image, prev, diff);
		image.copyTo(prev);

	} while (countNonZero(diff) > 0);

	image *= 255;
}

/*
 * Gets all descriptors for a thinned image, and returns
 *
 * @param input_thinned : Thinned out image of fingerprint
 * keypoints : Important corners within fingerprint image
 */
Mat getDescriptors(Mat& input_thinned, vector<KeyPoint> keypoints)
{
	Ptr<Feature2D> orb_descriptor = ORB::create();
	Mat descriptors;
	orb_descriptor->compute(input_thinned, keypoints, descriptors);

	return descriptors;
}

/*
 * Helper to run all thinning operations, and find important corners
 *
 * @param input : Image of fingerprint to be operated on
 * @param display : Whether to show all comparisions after applied operations
 * @return : Descriptors grabbed from getDescriptors()
 */
Mat applyAlgo(Mat &input, bool display = false)
{
	Mat input_binary;
	threshold(input, input_binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	
	// Apply thinning operation
	Mat input_thinned = input_binary.clone();
	thinning(input_thinned);
	
	// Find strong points within fingerprint
	Mat harris_corners, harris_normalised;
	harris_corners = Mat::zeros(input_thinned.size(), CV_32FC1);
	cv::cornerHarris(input_thinned, harris_corners, 2, 3, 0.04, BORDER_DEFAULT);
	cv::normalize(harris_corners, harris_normalised, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	int threshold_harris = 125;
	vector<KeyPoint> keypoints;

	Mat rescaled;
	cv::convertScaleAbs(harris_normalised, rescaled);
	Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
	Mat in[] = { rescaled, rescaled, rescaled };
	int from_to[] = { 0,0, 1,1, 2,2 };

	cv::mixChannels(in, 3, &harris_c, 1, from_to, 3);

	// Find all keypoints within fingerprint for comparisions
	for (int x = 0; x < harris_normalised.cols; x++) {
		for (int y = 0; y < harris_normalised.rows; y++) {
			if ((int)harris_normalised.at<float>(y, x) > threshold_harris) {
				circle(harris_c, Point(x, y), 5, Scalar(0, 255, 0), 1);
				circle(harris_c, Point(x, y), 1, Scalar(0, 0, 255), 1);
				keypoints.push_back(KeyPoint(x, y, 1));
			}
		}
	}
	
	// Display all comparisions
	if (display) {
		Mat container_1(input.rows, input.cols * 2, CV_8UC1);
		input.copyTo(container_1(Rect(0, 0, input.cols, input.rows)));
		input_binary.copyTo(container_1(Rect(input.cols, 0, input.cols, input.rows)));

		Mat container_2(input.rows, input.cols * 2, CV_8UC1);
		input_binary.copyTo(container_2(Rect(0, 0, input.cols, input.rows)));
		input_thinned.copyTo(container_2(Rect(input.cols, 0, input.cols, input.rows)));

		Mat container_3(input.rows, input.cols * 2, CV_8UC3);
		Mat input_thinned_c = input_thinned.clone();
		cv::cvtColor(input_thinned_c, input_thinned_c, COLOR_GRAY2RGB);
		input_thinned_c.copyTo(container_3(Rect(0, 0, input.cols, input.rows)));
		harris_c.copyTo(container_3(Rect(input.cols, 0, input.cols, input.rows)));

		imshow("Input VS Binary", container_1); waitKey(0);
		imshow("Binary VS Thinned", container_2); waitKey(0);
		imshow("Thinned VS Corners", container_3); waitKey(0);
	}

	return getDescriptors(input_thinned, keypoints);
}

/*
 * Start of program
 */
int main(int argc, const char** argv)
{
	string images[10] = {"101_2.tif", "101_3.tif", "101_4.tif", "101_5.tif", "101_6.tif", "101_7.tif", "101_8.tif", "102_1.tif", "102_2.tif", "102_3.tif"};
	map<string, float> scores;

	// Load 'control' fingerprint
	Mat input_1 = imread(CONTROL_IMG, IMREAD_GRAYSCALE);
	if (input_1.empty()) {
		cout << "Failed to load " << CONTROL_IMG << endl;
		return 1;
	}

	Mat descriptors_1 = applyAlgo(input_1);

	// Load random fingerprint & test against control fingerprint
	for (int i = 0; i < 3; i++) {
		int num = rand() % 10;

		Mat input_2 = imread(images[num], IMREAD_GRAYSCALE);
		if (input_2.empty()) {
			cout << "Failed to load " << images[num] << endl;
			return 1;
		}

		cout << "Testing " << CONTROL_IMG << " VS " << images[num];

		// Apply binarization & thinning operations
		Mat descriptors_2 = applyAlgo(input_2);

		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		vector<DMatch> matches;
		matcher->match(descriptors_1, descriptors_2, matches);

		float score = 0.0;
		for (int i = 0; i < matches.size(); i++) {
			score += matches[i].distance;
		}

		score /= matches.size();

		cout << " - " << ((score <= THRESHOLD) ? "MATCH" : "NO MATCH") << endl;
	}

	return 0;
}