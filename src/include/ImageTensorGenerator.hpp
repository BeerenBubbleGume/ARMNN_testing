#pragma once
#include "InferanceImage.hpp"

// Parameters used in normalizing images
struct NormalizationParameters
{
    float scale{ 1.0 };
    array<float, 3> mean{ { 0.0, 0.0, 0.0 } };
    array<float, 3> stddev{ { 1.0, 1.0, 1.0 } };
};

enum class SupportedFrontend
{
    TFLite     = 0,
};

/** Get normalization parameters.
 * Note that different flavours of models and different model data types have different normalization methods.
 * This tool currently only supports TF and TFLite models
 *
 * @param[in] modelFormat   One of the supported frontends
 * @param[in] outputType    Output type of the image tensor, also the type of the intended model
 */
NormalizationParameters GetNormalizationParameters(const SupportedFrontend& modelFormat,
                                                   const armnn::DataType& outputType)
{
    NormalizationParameters normParams;
    // Explicitly set default parameters
    normParams.scale  = 1.0;
    normParams.mean   = { 0.0, 0.0, 0.0 };
    normParams.stddev = { 1.0, 1.0, 1.0 };
    switch (modelFormat)
    {
        case SupportedFrontend::TFLite:
        default:
            switch (outputType)
            {
                case armnn::DataType::Float32:
                    normParams.scale = 127.5;
                    normParams.mean  = { 1.0, 1.0, 1.0 };
                    break;
                case armnn::DataType::Signed32:
                    normParams.mean = { 128.0, 128.0, 128.0 };
                    break;
                case armnn::DataType::QAsymmU8:
                    break;
                case armnn::DataType::QAsymmS8:
                    normParams.mean = { 128.0, 128.0, 128.0 };
                    break;
                default:
                    break;
            }
            break;
    }
    return normParams;
}

/** Prepare raw image tensor data by loading the image from imagePath and preprocessing it.
 *
 * @param[in] imagePath     Path to the image file
 * @param[in] newWidth      The new width of the output image tensor
 * @param[in] newHeight     The new height of the output image tensor
 * @param[in] normParams    Normalization parameters for the normalization of the image
 * @param[in] batchSize     Batch size
 * @param[in] outputLayout  Data layout of the output image tensor
 */
template <typename ElemType>
vector<ElemType> PrepareImageTensor(const string& imagePath,
                                         unsigned int newWidth,
                                         unsigned int newHeight,
                                         const NormalizationParameters& normParams,
                                         unsigned int batchSize                = 1,
                                         const armnn::DataLayout& outputLayout = armnn::DataLayout::NHWC);

// Prepare float32 image tensor
template <>
vector<float> PrepareImageTensor<float>(const string& imagePath,
                                             unsigned int newWidth,
                                             unsigned int newHeight,
                                             const NormalizationParameters& normParams,
                                             unsigned int batchSize,
                                             const armnn::DataLayout& outputLayout)
{
    // Generate image tensor
    vector<float> imageData;
    InferenceTestImage testImage(imagePath.c_str());
    if (newWidth == 0)
    {
        newWidth = testImage.GetWidth();
    }
    if (newHeight == 0)
    {
        newHeight = testImage.GetHeight();
    }
    // Resize the image to new width and height or keep at original dimensions if the new width and height are specified
    // as 0 Centre/Normalise the image.
    imageData = testImage.Resize(newWidth, newHeight, CHECK_LOCATION(),
                                 InferenceTestImage::ResizingMethods::BilinearAndNormalized, normParams.mean,
                                 normParams.stddev, normParams.scale);
    if (outputLayout == armnn::DataLayout::NCHW)
    {
        // Convert to NCHW format
        const armnn::PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
        armnn::TensorShape dstShape({ batchSize, 3, newHeight, newWidth });
        vector<float> tempImage(imageData.size());
        armnnUtils::Permute(dstShape, NHWCToArmNN, imageData.data(), tempImage.data(), sizeof(float));
        imageData.swap(tempImage);
    }
    return imageData;
}

// Prepare int32 image tensor
template <>
vector<int> PrepareImageTensor<int>(const string& imagePath,
                                         unsigned int newWidth,
                                         unsigned int newHeight,
                                         const NormalizationParameters& normParams,
                                         unsigned int batchSize,
                                         const armnn::DataLayout& outputLayout)
{
    // Get float32 image tensor
    vector<float> imageDataFloat =
        PrepareImageTensor<float>(imagePath, newWidth, newHeight, normParams, batchSize, outputLayout);
    // Convert to int32 image tensor with static cast
    vector<int> imageDataInt;
    imageDataInt.reserve(imageDataFloat.size());
    transform(imageDataFloat.begin(), imageDataFloat.end(), back_inserter(imageDataInt),
                   [](float val) { return static_cast<int>(val); });
    return imageDataInt;
}

// Prepare qasymmu8 image tensor
template <>
vector<uint8_t> PrepareImageTensor<uint8_t>(const string& imagePath,
                                                 unsigned int newWidth,
                                                 unsigned int newHeight,
                                                 const NormalizationParameters& normParams,
                                                 unsigned int batchSize,
                                                 const armnn::DataLayout& outputLayout)
{
    // Get float32 image tensor
    vector<float> imageDataFloat =
        PrepareImageTensor<float>(imagePath, newWidth, newHeight, normParams, batchSize, outputLayout);
    vector<uint8_t> imageDataQasymm8;
    imageDataQasymm8.reserve(imageDataFloat.size());
    // Convert to uint8 image tensor with static cast
    std::transform(imageDataFloat.begin(), imageDataFloat.end(), std::back_inserter(imageDataQasymm8),
                   [](float val) { return static_cast<uint8_t>(val); });
    return imageDataQasymm8;
}

// Prepare qasymms8 image tensor
template <>
vector<int8_t> PrepareImageTensor<int8_t>(const string& imagePath,
                                               unsigned int newWidth,
                                               unsigned int newHeight,
                                               const NormalizationParameters& normParams,
                                               unsigned int batchSize,
                                               const armnn::DataLayout& outputLayout)
{
    // Get float32 image tensor
    vector<float> imageDataFloat =
            PrepareImageTensor<float>(imagePath, newWidth, newHeight, normParams, batchSize, outputLayout);
    vector<int8_t> imageDataQasymms8;
    imageDataQasymms8.reserve(imageDataFloat.size());
    // Convert to uint8 image tensor with static cast
    std::transform(imageDataFloat.begin(), imageDataFloat.end(), std::back_inserter(imageDataQasymms8),
                   [](float val) { return static_cast<uint8_t>(val); });
    return imageDataQasymms8;
}

/** Write image tensor to ofstream
 *
 * @param[in] imageData         Image tensor data
 * @param[in] imageTensorFile   Output filestream (ofstream) to which the image tensor data is written
 */
template <typename ElemType>
void WriteImageTensorImpl(const vector<ElemType>& imageData, ofstream& imageTensorFile)
{
    std::copy(imageData.begin(), imageData.end(), std::ostream_iterator<ElemType>(imageTensorFile, " "));
}

// For uint8_t image tensor, cast it to int before writing it to prevent writing data as characters instead of
// numerical values
template <>
void WriteImageTensorImpl<uint8_t>(const vector<uint8_t>& imageData, ofstream& imageTensorFile)
{
    std::copy(imageData.begin(), imageData.end(), std::ostream_iterator<int>(imageTensorFile, " "));
}

// For int8_t image tensor, cast it to int before writing it to prevent writing data as characters instead of
// numerical values
template <>
void WriteImageTensorImpl<int8_t>(const vector<int8_t>& imageData, ofstream& imageTensorFile)
{
    std::copy(imageData.begin(), imageData.end(), std::ostream_iterator<int>(imageTensorFile, " "));
}