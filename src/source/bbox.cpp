#include "../include/bbox.hpp"

nc::NdArray<float> bboxes::yolo_correct_boxes(nc::NdArray<float>  box_xy, nc::NdArray<float>  box_wh, 
                                vector<float> input_shape, vector<float> image_shape){
    auto boxYX = box_xy(box_xy.rSlice(), nc::Slice(-1));
    auto boxHW = box_wh(box_wh.rSlice(), nc::Slice(-1));
    auto inputShape = nc::NdArray<float>(input_shape);
    auto imageShape = nc::NdArray<float>(image_shape);
    auto newShape = nc::round(imageShape * nc::min(inputShape / imageShape));
    nc::NdArray<float> offset;
    nc::NdArray<float> scale;

    for (int i = 0; i < static_cast<int>(inputShape.size()); ++i){
        offset[i] = (inputShape[i] - newShape[i]) / 2.f / inputShape[i];
        scale[i] = inputShape[i] / newShape[i];
    }

    boxYX = (box_xy - offset) * scale;
    boxHW *= scale;

    auto boxMinis = boxYX - (boxHW / 2.f);
    auto boxMaxes = boxYX + (boxHW / 2.f);
    auto listBoxPoints = {boxMinis(boxMinis.rSlice(), nc::Slice(0, 1)), 
                                    boxMinis(boxMinis.rSlice(), nc::Slice(1, 2)),
                                    boxMaxes(boxMaxes.rSlice(), nc::Slice(0, 1)),
                                    boxMaxes(boxMaxes.rSlice(), nc::Slice(1, 2))};
    auto boxes = nc::concatenate(listBoxPoints, nc::Axis(-1));
    auto _imageShape = {imageShape, nc::NdArray<float>(image_shape.begin(), image_shape.end())};
    boxes *= nc::concatenate(_imageShape, nc::Axis(-1));
    return boxes;
}

vector<nc::NdArray<float>> bboxes::preprocess(nc::NdArray<float> output, vector<float> image_data, vector<float> image_shape){
    auto boxXY = (output(output.rSlice(), {0, 2}) + output(output.rSlice(), {2, 4})) / 2.f;
    auto boxWH = output(output.rSlice(), {2, 4}) - output(output.rSlice(), {0, 2});

    output(output.rSlice(), output.rSlice(4)) = yolo_correct_boxes(boxXY, boxWH, image_data, image_shape);
    vector<nc::NdArray<float>> ret = {output(output.rSlice(), output.rSlice(4)), output(output.rSlice(), 
                                        output.rSlice(4)) * output(output.rSlice(),output.rSlice(5)),
                                        nc::NdArray<float>(output(output.rSlice(), output.rSlice(6)))};
    
    return ret;
}

bboxes::bboxes(){

}
bboxes::~bboxes(){

}