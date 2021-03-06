/*
 * Names : Zachary Sluder, Stacey Lawson, Elijah Meyer, Alexander Miller, Alton Panton
 * Course : CS-4410 Software Engineering
 *
 * Implements OpenCV to compare a fingerprint image against a 'database'
 */
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

// Sample fingerprint image that is comapred with all other images
string CONTROL_FILENAME = "101_6.tif";

// Threshold for fingerprint comparisions. EX: result <= THRESHOLD is a match
#define THRESHOLD 50         

// Number of failed attempts to test fingerprint
#define FAILED_ATTEMPTS 3      

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
 * @param filename : Used to save thinned file for later use
 * @param apply_thinning : Whether to apply thinning operation on fingerprint
 * @param display : Whether to show all comparisions after applied operations
 * @return : Descriptors grabbed from getDescriptors()
 */
Mat applyAlgo(Mat& input, string filename, bool apply_thinning = true, bool display = false)
{
	Mat input_binary, input_thinned, harris_corners, harris_normalised, rescaled;
	threshold(input, input_binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	
	// Apply thinning operation
	if (apply_thinning) {
		input_thinned = input_binary.clone();
		thinning(input_thinned);

		imwrite("thinned_" + filename, input_thinned);
	} else {
		input_thinned = input;
	}
	
	// Find strong points within fingerprint
	harris_corners = Mat::zeros(input_thinned.size(), CV_32FC1);
	cv::cornerHarris(input_thinned, harris_corners, 2, 3, 0.04, BORDER_DEFAULT);
	cv::normalize(harris_corners, harris_normalised, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	vector<KeyPoint> keypoints;

	cv::convertScaleAbs(harris_normalised, rescaled);
	Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
	Mat in[] = {rescaled, rescaled, rescaled};
	int from_to[] = {0, 0, 1, 1, 2, 2};

	cv::mixChannels(in, 3, &harris_c, 1, from_to, 3);

	// Find all keypoints within fingerprint for comparisions
	for (int x = 0; x < harris_normalised.cols; x++) {
		for (int y = 0; y < harris_normalised.rows; y++) {
			if ((int)harris_normalised.at<float>(y, x) > 125) {
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
	string images[10] = {"101_2.tif", "102_8.tif", "101_4.tif", "101_5.tif", "101_6.tif"};
	map<string, float> scores;

	int wrong_counter = 0;

	// Load 'control' fingerprint
	Mat input_1 = imread(CONTROL_FILENAME, IMREAD_GRAYSCALE);
	if (input_1.empty()) {
		cout << "Failed to load " << CONTROL_FILENAME << endl;

		return EXIT_FAILURE;
	}

	Mat descriptors_1 = applyAlgo(input_1, CONTROL_FILENAME);
	
	// Load random fingerprint & test against control fingerprint
	while (wrong_counter < FAILED_ATTEMPTS) {
		int num = rand() % 5;
		string filename = images[num];

		cout << "Testing " << CONTROL_FILENAME << " VS " << filename;

		// Check if already thinned file exists 
		ifstream infile("thinned_" + filename);
		Mat input_2, descriptors_2;

		if (infile.good()) {
			input_2 = imread("thinned_" + filename, IMREAD_GRAYSCALE);
			descriptors_2 = applyAlgo(input_2, filename, false);
		} else {
			input_2 = imread(filename, IMREAD_GRAYSCALE);
			descriptors_2 = applyAlgo(input_2, filename);
		}

		if (input_2.empty()) {
			cout << "Failed to load " << filename << endl;

			return EXIT_FAILURE;
		}

		// Compare descriptors and calculate final score
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
		vector<DMatch> matches;
		matcher->match(descriptors_1, descriptors_2, matches);

		float score = 0.0;
		for (int i = 0; i < matches.size(); i++) {
			score += matches[i].distance;
		}

		score /= matches.size();

		if (score <= THRESHOLD) {
			cout << " - " << score << " - " << "MATCH" << endl;

			return EXIT_SUCCESS;
		} else {
			cout << " - " << score << " - " << "NO MATCH" << endl;
			wrong_counter++;
		}
	}

	return EXIT_FAILURE;
}