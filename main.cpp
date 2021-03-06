#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include "tf_utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <windows.h>

#define IMG_SIZE 784
#define DATA_SIZE 500

static int displayGraphInfo(char* filepath)
{
	TF_Graph *graph = tf_utils::LoadGraphDef(filepath);
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}

	size_t pos = 0;
	TF_Operation* oper;
	printf("--- graph info ---\n");
	while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
		printf("%s\n", TF_OperationName(oper));
	}
	printf("--- graph info ---\n");

	TF_DeleteGraph(graph);
	return 0;
}

void getCurrentPath(char* directory)
{
	char buffer[512];
	GetModuleFileNameA(NULL, buffer, 512); // get the DLL path
	char* pos = buffer;
	while (1) {
		char* p = strchr(pos, '\\');
		if (p == NULL) break;
		pos = ++p;
	}
	strncpy(directory, buffer, pos - buffer - 1);
	directory[pos - buffer - 1] = '\0';
}

std::vector<std::string> split(std::string str, char del)
{
	std::vector<std::string> result;
	std::string subStr;

	for (const char c : str) {
		if (c == del) {
			result.push_back(subStr);
			subStr.clear();
		}
		else {
			subStr += c;
		}
	}

	result.push_back(subStr);
	return result;
}

bool readImage(char* filepath, std::vector<std::vector<float>>& images)
{
	std::ifstream ifs(filepath);
	if (!ifs.is_open())
	{
		return false;
	}

	for (int i = 0; i < DATA_SIZE; i++)
	{
		std::string line;
		getline(ifs, line);
		std::vector<std::string> vals = split(line, ',');
		for (int j = 0; j < IMG_SIZE; j++)
		{
			images[i][j] = std::stof(vals[j + 1])/255.0;
		}
	}

	ifs.close();

	return true;
}

void showImage(std::vector<std::vector<float>>& images, int img_num)
{
	int size = sqrt(IMG_SIZE);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			double val = images[img_num][i * 28 + j];
			if (val < 0.5) printf("#");
			else printf("*");
		}
		printf("\n");
	}
}

int main()
{
	printf("Hello from TensorFlow C library version %s\n", TF_Version());

	char directory[512];
	getCurrentPath(directory);

	char image_path[512];
	strcpy(image_path, directory);
	strcat(image_path, "\\testImage.txt");
	std::vector<std::vector<float>> images(DATA_SIZE, std::vector<float>(IMG_SIZE, 0.0));
	if (!readImage(image_path, images))
	{
		std::cout << "Cannot read images" << std::endl;
		return -1;
	}

	/* get graph info */
	char model_path[512];
	strcpy(model_path, directory);
	strcat(model_path, "\\frozen_graph.pb");
	displayGraphInfo(model_path);

	TF_Graph *graph = tf_utils::LoadGraphDef(model_path);
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}

	/* prepare input tensor */
	TF_Output input_op = { TF_GraphOperationByName(graph, "conv2d_1_input"), 0 };
	if (input_op.oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}

	TF_Tensor* output_tensor = nullptr;

	/* prepare session */
	TF_Status* status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, options, status);
	TF_DeleteSessionOptions(options);

	if (TF_GetCode(status) != TF_OK) {
		TF_DeleteStatus(status);
		return 4;
	}

	const std::vector<std::int64_t> input_dims = { 1, 28, 28, 1 };
	std::vector<float> input_vals(IMG_SIZE);
	for (int j = 0; j < DATA_SIZE; j++)
	{
		for (int i = 0; i < IMG_SIZE; i++)
		{
			input_vals[i] = images[j][i];
		}

		TF_Tensor* input_tensor = tf_utils::CreateTensor(TF_FLOAT,
			input_dims.data(), input_dims.size(),
			input_vals.data(), input_vals.size() * sizeof(float));

		/* prepare output tensor */
		TF_Output out_op = { TF_GraphOperationByName(graph, "dense_2/Softmax"), 0 };
		if (out_op.oper == nullptr) {
			std::cout << "Can't init out_op" << std::endl;
			return 3;
		}

		/* run session */
		TF_SessionRun(sess,
			nullptr, // Run options.
			&input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
			&out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
			nullptr, 0, // Target operations, number of targets.
			nullptr, // Run metadata.
			status // Output status.
		);

		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Error run session";
			TF_DeleteStatus(status);
			return 5;
		}

		showImage(images, j);

		const auto probs = static_cast<float*>(TF_TensorData(output_tensor));
		for (int i = 0; i < 10; i++) {
			printf("prob of %d: %.3f\n", i, probs[i]);
		}

		TF_DeleteTensor(input_tensor);

		printf("\nEnter any key->");
		getchar();
	}

	TF_CloseSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error close session";
		TF_DeleteStatus(status);
		return 6;
	}

	TF_DeleteSession(sess, status);
	if (TF_GetCode(status) != TF_OK) {
		std::cout << "Error delete session";
		TF_DeleteStatus(status);
		return 7;
	}
	
	TF_DeleteTensor(output_tensor);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);

	getchar();

	return 0;
}

