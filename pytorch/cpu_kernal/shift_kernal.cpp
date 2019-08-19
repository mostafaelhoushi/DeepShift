#include <torch/extension.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <vector>
using namespace std;
 torch::Tensor linear_kernal(
    torch::Tensor input,
    torch::Tensor shift,
    torch::Tensor sign ,
    torch::Tensor bias)
{
    cout<<"batch: "<<input.size(0)<<endl;
    cout<<"input feature: "<<input.size(1)<<endl;
    cout<<"shift input feature: "<<shift.size(0)<<endl;
    cout<<"shift output feature: "<<shift.size(1)<<endl;
    torch::Tensor output = torch::zeros({input.size(0),shift.size(0)}, torch::dtype(torch::kInt32));
     for( int batch = 0 ;  batch < input.size(0); batch++){
        for(int output_feature = 0 ; output_feature < shift.size(0); output_feature++){
            for(int input_feature = 0; input_feature <input.size(1);input_feature++){
                // cout<<"0"<<endl;
                auto s = shift[output_feature][input_feature].item<int8_t>();
                // cout<<"1"<<endl;
                auto y = output[batch][output_feature].item<int32_t>();
                // cout<<"2"<<endl;
                auto x = input[batch][input_feature].item<int32_t>();
                // cout<<"3"<<endl;
                if(sign[output_feature][input_feature].item<bool>()){
                    y -= (x << s);
                }
                else{
                    y += (x << s);
                }
                output[batch][output_feature] = y;
                // cout<<"4"<<endl;
            }
            auto b = bias[output_feature].item<int32_t>();
            // cout<<"5"<<endl;
            output[batch][output_feature] += b;
            cout<<"6"<<endl;
        }   
     }
   
    return output;
}   

torch::Tensor convolution_kernal(
    torch::Tensor input_,
    torch::Tensor shift,
    torch::Tensor sign ,
    torch::Tensor bias,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding)
{
    int strides_h;
    int strides_w;
    if(strides.size() ==1){
        strides_h = strides[0];
        strides_w = strides[0];
    }
    else{
        strides_h = strides[0];
        strides_w = strides[1];
    }
    torch::Tensor input = torch::constant_pad_nd(input_,padding, 0);
    int out_height = (input.size(2) - shift.size(2)) / strides_h +1;
    
    int out_width = (input.size(3) - shift.size(3)) / strides_w +1;
    
    torch::Tensor output = torch::zeros({input.size(0),shift.size(0),out_height, out_width }, torch::dtype(torch::kInt32));
    cout<<output<<endl;
    for (int batch = 0; batch < output.size(0); batch++) {//batch
		for (int out_channel = 0; out_channel < output.size(1); out_channel++) {//out_channel
            auto b = bias[out_channel].item<int32_t>();
			for (int out_height = 0; out_height < output.size(2); out_height++) {//out_height
				for (int out_width = 0; out_width < output.size(3); out_width++) {//out_width
					for (int filter_height = 0; filter_height < shift.size(2); filter_height++) {//filter_height
						for (int filter_width = 0; filter_width < shift.size(3); filter_width++) {//filter_width
							for (int in_channel = 0; in_channel < input.size(1); in_channel++) {//in_channel
                                
                                auto s = shift[out_channel][in_channel][filter_height][filter_width].item<int8_t>();
                              
                                auto out = output[batch][out_channel][out_height][out_width].item<int32_t>();
                              
                                auto data = input[batch][in_channel][out_height * strides_h + filter_height][out_width * strides_w + filter_width].item<int32_t>();
                               
                                if(sign[out_channel][in_channel][filter_height][filter_width].item<bool>()){
                                    out -= (data << s);
                                }
                                else{
                                    out += (data << s);
                                }
                    
								output[batch][out_channel][out_height][out_width] =out;

							}
						}
					}
                 
                    output[batch][out_channel][out_height][out_width] +=b;
				}
			}
            
            
		}
	}
    return output;
}

PYBIND11_MODULE(shift_kernal, m) {
    m.def("linear_kernal", &linear_kernal, "linear_kernal");
    m.def("convolution_kernal", &convolution_kernal, "convolution_kernal");
}

