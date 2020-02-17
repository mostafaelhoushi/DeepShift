#include <torch/extension.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <vector>
#include <ctime>
#include <thread>

#include "ATen/ATen.h"

#include "ATen/NativeFunctions.h"

#include <ATen/Parallel.h>
// #include <omp.h>
using namespace std;
#define MAX_THREAD 10

//  torch::Tensor linear_kernal(
//     torch::Tensor input,
//     torch::Tensor shift,
//     torch::Tensor sign ,
//     torch::Tensor bias)
// {
//     cout<<"batch: "<<input.size(0)<<endl;
//     cout<<"input feature: "<<input.size(1)<<endl;
//     cout<<"shift input feature: "<<shift.size(0)<<endl;
//     cout<<"shift output feature: "<<shift.size(1)<<endl;
//     std::clock_t start;
//     double duration;
//     // start = std::clock();
//     torch::Tensor output = torch::zeros({input.size(0),shift.size(0)}, torch::dtype(torch::kInt32));
//     // #pragma omp parallel num_threads(20)
//      for( int batch = 0 ;  batch < input.size(0); batch++){
//          cout<<"batch: "<<batch<<endl;
//          start = std::clock();
//         for(int output_feature = 0 ; output_feature < shift.size(0); output_feature++){
//             for(int input_feature = 0; input_feature <input.size(1);input_feature++){
//                 // cout<<"0"<<endl;
//                 auto s = shift[output_feature][input_feature].item<int8_t>();
//                 // cout<<"1"<<endl;
//                 auto y = output[batch][output_feature].item<int32_t>();
//                 // cout<<"2"<<endl;
//                 auto x = input[batch][input_feature].item<int32_t>();
//                 // cout<<"3"<<endl;
//                 if(sign[output_feature][input_feature].item<bool>()){
//                     y -= (x << s);
//                 }
//                 else{
//                     y += (x << s);
//                 }
//                 output[batch][output_feature] = y;
//                 // cout<<"4"<<endl;
//             }
//             auto b = bias[output_feature].item<int32_t>();
//             // cout<<"5"<<endl;
//             output[batch][output_feature] += b;
//             // cout<<"6"<<endl;
//         }
//         duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

//         std::cout<<"One batch use "<< duration <<'\n';
   
//      }
//     // std::cout<<"Finish one call"<<std::endl;
//     // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

//     // std::cout<<"One call use "<< duration <<'\n';
//     return output;
// }   
void stub(
    vector<vector<int32_t>>& input,
    vector<vector<int8_t>>& shift,
    vector<vector<int32_t>>& sign ,
    vector<int32_t>& bias,
    unsigned int start, 
    unsigned int end,
    unsigned int idx,
    vector<vector<int32_t>>& output)
{
    
    for(unsigned int  batch = 0 ;  batch < input.size(); batch++    ){
        for(unsigned int output_feature = start ; output_feature < end; output_feature++){
            for(unsigned int input_feature = 0; input_feature <input[0].size();input_feature++){
                auto s = shift[output_feature][input_feature];
                auto y = output[batch][output_feature];
                auto x = input[batch][input_feature];
                if(sign[output_feature][input_feature] < 0){
                    y -= (x << s);
                }
                else if(sign[output_feature][input_feature] > 0) {
                    y += (x << s);
                }
                //else if(sign[output_feature][input_feature] == 0) {
                //    y += 0;
                //}
              
                output[batch][output_feature] = y;
             
            }
            auto b = bias[output_feature];
      
            output[batch][output_feature] += b;
     
            
        }
    }
    
}

vector<vector<int32_t>> linear_kernel(
    vector<vector<int32_t>>& input,
    vector<vector<int8_t>>& shift,
    vector<vector<int32_t>>& sign ,
    vector<int32_t>& bias)
{
    // for(auto a :sign){
    //     for(auto b : a){
    //         cout<<b<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<"batch: "<<input.size()<<endl;
    // cout<<"input feature: "<<input[0].size()<<endl;
    // cout<<"shift output feature: "<<shift.size()<<endl;
    // cout<<"shift input feature: "<<shift[0].size()<<endl;
    // std::clock_t start;
    // double duration;
    // start = std::clock();
   
    vector<int32_t> n(shift.size(), 0); 
    vector<vector<int32_t>> output(input.size(), n);

    // //**************************************
    // // vector<int32_t> temp1(shift.size()/MAX_THREAD, 0);
    // // vector<int32_t> temp2(shift.size()/MAX_THREAD, 0);
    // // vector<vector<int32_t>> vv1(input.size(), temp1);
    // // vector<vector<int32_t>> vv2(input.size(), temp1);
    // vector<thread> tp(MAX_THREAD);
    // int work_load = shift.size() / MAX_THREAD;

    // for( unsigned int i = 0 ; i < MAX_THREAD; i++){
    //     if(i != MAX_THREAD - 1){
    //         tp[i] = thread(stub, std::ref(input), std::ref(shift), std::ref(sign), std::ref(bias), i*work_load, (i + 1)* work_load, i, std::ref(output));
    //     }
    //     else{
    //         tp[i] = thread(stub, std::ref(input), std::ref(shift), std::ref(sign), std::ref(bias), i*work_load, shift.size(), i, std::ref(output));
    //     }
    // }

    // for( unsigned int i = 0 ; i < MAX_THREAD; i++){
    //     if(tp[i].joinable()){
    //         tp[i].join();
    //     }
    // }
    // // cout<<"here!!!!!!!!!"<<endl;
    // // for(unsigned int i = 0 ; i < vv1.size(); i++){
    // //     for(int j = 0; j < work_load; j++){
    // //         output.at(i).at(j)= vv1.at(i).at(j);
    // //     }
    // // }
    // // cout<<"there!!!!!!!!!"<<endl;
    // // for(unsigned int i = 0 ; i < vv2.size(); i++){
    // //     for(int j = 0; j < work_load; j++){
    // //         output.at(i).at(j+work_load)= vv2.at(i).at(j);
    // //     }
    // // }
    // // cout<<"end!!!!!!!!!"<<endl;
    // // cout<<vv1.size()<<endl;
    // // cout<<vv1[0].size()<<endl;
    // // cout<<vv2.size()<<endl;
    // // cout<<vv2[0].size()<<endl;
    // // cout<<work_load<<endl;


    //****************************************
    // #pragma omp parallel num_threads(10)
    at::parallel_for(0, input.size(), 10, [&](int64_t start, int64_t end){
    //  for( unsigned int  batch = 0 ;  batch < input.size(); batch++){
        for( auto batch = start ;  batch < end; batch++){
        //  cout<<"batch: "<<batch<<endl;
        //  start = std::clock();
        for(unsigned int output_feature = 0 ; output_feature < shift.size(); output_feature++){
            for(unsigned int input_feature = 0; input_feature <input[0].size();input_feature++){
                // cout<<"0"<<endl;
                auto s = shift[output_feature][input_feature];
                // cout<<"1"<<endl;
                auto y = output[batch][output_feature];
                // cout<<"2"<<endl;
                auto x = input[batch][input_feature];
                // cout<<"3"<<endl;
                if(sign[output_feature][input_feature] < 0 && s >=0 ){
                    y -= (x << s);
                }
                else if(sign[output_feature][input_feature] > 0 && s >=0){
                    y += (x << s);
                }
                else if(sign[output_feature][input_feature] < 0 && s <0){
                    y -= (x >> (-s));
                }
                else if(sign[output_feature][input_feature] > 0 && s <0) {
                    y += (x >> (-s));
                }
                output[batch][output_feature] = y;
                // if(batch == 0 && output_feature == 1){
                //     cout<<"data: "<<x;
                //     cout<<"  shift: "<<(int32_t)s;
                //     cout<<"  sign: "<<sign[output_feature][input_feature];
                //     cout<<" output: "<<y<<endl;
                // }
                // cout<<"4"<<endl;
            }
            auto b = bias[output_feature];
            // cout<<"5"<<endl;
            output[batch][output_feature] += b;
            // cout<<"6"<<endl;
        }
        // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        // std::cout<<"One batch use "<< duration <<'\n';
   
     }
    });


    // std::cout<<"Finish one call"<<std::endl;
    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    // std::cout<<"One call use "<< duration <<'\n';
    return output;
}   


// torch::Tensor convolution_kernal(
//     torch::Tensor input_,
//     torch::Tensor shift,
//     torch::Tensor sign ,
//     torch::Tensor bias,
//     torch::IntArrayRef strides,
//     torch::IntArrayRef padding)
// {
//     int strides_h;
//     int strides_w;
//     if(strides.size() ==1){
//         strides_h = strides[0];
//         strides_w = strides[0];
//     }
//     else{
//         strides_h = strides[0];
//         strides_w = strides[1];
//     }
//     torch::Tensor input = torch::constant_pad_nd(input_,padding, 0);
//     int out_height = (input.size(2) - shift.size(2)) / strides_h +1;
    
//     int out_width = (input.size(3) - shift.size(3)) / strides_w +1;
    
//     torch::Tensor output = torch::zeros({input.size(0),shift.size(0),out_height, out_width }, torch::dtype(torch::kInt32));
//     cout<<output<<endl;
//     for (int batch = 0; batch < output.size(0); batch++) {//batch
// 		for (int out_channel = 0; out_channel < output.size(1); out_channel++) {//out_channel
//             auto b = bias[out_channel].item<int32_t>();
// 			for (int out_height = 0; out_height < output.size(2); out_height++) {//out_height
// 				for (int out_width = 0; out_width < output.size(3); out_width++) {//out_width
// 					for (int filter_height = 0; filter_height < shift.size(2); filter_height++) {//filter_height
// 						for (int filter_width = 0; filter_width < shift.size(3); filter_width++) {//filter_width
// 							for (int in_channel = 0; in_channel < input.size(1); in_channel++) {//in_channel
                                
//                                 auto s = shift[out_channel][in_channel][filter_height][filter_width].item<int8_t>();
                              
//                                 auto out = output[batch][out_channel][out_height][out_width].item<int32_t>();
                              
//                                 auto data = input[batch][in_channel][out_height * strides_h + filter_height][out_width * strides_w + filter_width].item<int32_t>();
                               
//                                 if(sign[out_channel][in_channel][filter_height][filter_width].item<bool>()){
//                                     out -= (data << s);
//                                 }
//                                 else{
//                                     out += (data << s);
//                                 }
                    
// 								output[batch][out_channel][out_height][out_width] =out;

// 							}
// 						}
// 					}
                 
//                     output[batch][out_channel][out_height][out_width] +=b;
// 				}
// 			}
            
            
// 		}
// 	}
//     return output;
// }

vector<vector<vector<vector<int32_t>>>> convolution_kernel(
    vector<vector<vector<vector<int32_t>>>>& input,
    vector<vector<vector<vector<int8_t>>>>& shift,
    vector<vector<vector<vector<int32_t>>>>& sign ,
    vector<int32_t>& bias,
    torch::IntArrayRef strides,
    torch::IntArrayRef padding)
{
    // std::clock_t start;
    // double duration;
    // start = std::clock();
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
    // torch::Tensor input = torch::constant_pad_nd(input_,padding, 0);
    int out_height = (input[0][0].size() - shift[0][0].size()) / strides_h +1;
    
    int out_width = (input[0][0][0].size() - shift[0][0][0].size()) / strides_w +1;
    
    // torch::Tensor output = torch::zeros({input.size(),shift.size(),out_height, out_width }, torch::dtype(torch::kInt32));
    vector<int32_t> l1(out_width,0); 
    vector<vector<int32_t>> l2(out_height,l1);
    vector<vector<vector<int32_t>>> l3(shift.size(), l2);
    vector<vector<vector<vector<int32_t>>>> output(input.size(), l3);
    at::parallel_for(0, output.size(), 0, [&](int64_t start, int64_t end){
    for (unsigned int batch = start; batch < end; batch++) {//batch
		for (unsigned int out_channel = 0; out_channel < output[0].size(); out_channel++) {//out_channel
            auto b = bias[out_channel];
			for (unsigned int out_height = 0; out_height < output[0][0].size(); out_height++) {//out_height
				for (unsigned int out_width = 0; out_width < output[0][0][0].size(); out_width++) {//out_width
					for (unsigned int filter_height = 0; filter_height < shift[0][0].size(); filter_height++) {//filter_height
						for (unsigned int filter_width = 0; filter_width < shift[0][0][0].size(); filter_width++) {//filter_width
							for (unsigned int in_channel = 0; in_channel < input[0].size(); in_channel++) {//in_channel
                                
                                auto s = shift[out_channel][in_channel][filter_height][filter_width];
                              
                                auto y = output[batch][out_channel][out_height][out_width];
                              
                                auto x = input[batch][in_channel][out_height * strides_h + filter_height][out_width * strides_w + filter_width];
                               
                                if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s >=0 ){
                                    y -= (x << s);
                                }
                                else if(sign[out_channel][in_channel][filter_height][filter_width] > 0 && s >=0){
                                    y += (x << s);
                                }
                                else if(sign[out_channel][in_channel][filter_height][filter_width] < 0 && s <0){
                                    y -= (x >> (-s));
                                }
                                else{
                                    y += (x >> (-s));
                                }
								output[batch][out_channel][out_height][out_width] =y;

							}
						}
					}
                 
                    output[batch][out_channel][out_height][out_width] +=b;
				}
			}
            
            
		}
	}
    });

    // std::cout<<"Finish one call"<<std::endl;
    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    // std::cout<<"One call use "<< duration <<'\n';
    return output;
}

PYBIND11_MODULE(deepshift_cpu, m) {
    m.def("linear_kernel", &linear_kernel, "linear_kernel");
    m.def("convolution_kernel", &convolution_kernel, "convolution_kernel");
}

