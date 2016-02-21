#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_loss_layer_dense.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastiveLossDenseLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  dist_sq_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ContrastiveLossDenseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
  	for (int w = 0; w < width; ++w) {
  		for (int h = 0; h < height; ++h) {
			
			int idx1 = ((i*height)+h)*width+w;
			
			// calc distance between a and b
			Dtype dist_sq = 0;
			for (int c = 0; c < channels; ++c) {
				Dtype diff = *(diff_.cpu_data() + ((i*channels+c)*height+h)*width+w);
		    	dist_sq += diff*diff;//caffe_cpu_dot(1, tmp, tmp);
		    }
			dist_sq_.mutable_cpu_data()[idx1] = dist_sq;
    		
			// add to loss
    		if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
    		  loss += dist_sq_.cpu_data()[idx1];
    		} else {  // dissimilar pairs
    		  Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq_.cpu_data()[idx1]),Dtype(0.0));
    		  loss += dist*dist;
    		}
	  }
	}
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossDenseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  
  for (int i = 0; i < 2; ++i) { // iterates over the three bottoms
    if (propagate_down[i]) {
	  
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
	  int height = bottom[i]->height();
	  int width = bottom[i]->width();
      int channels = bottom[i]->channels();
	  
	  Dtype* bout = bottom[i]->mutable_cpu_diff();
	  
      for (int j = 0; j < num; ++j) {
	    for (int w = 0; w < width; ++w) {
	      for (int h = 0; h < height; ++h) {
			  for (int c = 0; c < channels; ++c) {
			    
				int idx1 = ((j*height)+h)*width+w;
				int idx2 = ((j*channels+c)*height+h)*width+w;
		    	
            	if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
            	  caffe_cpu_axpby(
            	      1,
            	      alpha,
            	      diff_.cpu_data() + idx2,
            	      Dtype(0.0),
            	      bout + idx2);
            	}
				
		    	else {  // dissimilar pairs
            	  Dtype mdist(0.0);
            	  Dtype beta(0.0);
		    	  
            	  Dtype dist = sqrt(dist_sq_.cpu_data()[idx1]);
            	  mdist = margin - dist;
            	  beta = -alpha * mdist / (dist + Dtype(1e-4));
            	  
            	  if (mdist > Dtype(0.0)) {
            	    caffe_cpu_axpby(
            	        1,
            	        beta,
            	        diff_.cpu_data() + idx2,
            	        Dtype(0.0),
            	        bout + idx2);
            	  } else {
            	    caffe_set(1, Dtype(0), bout + idx2);
            	  }
		  	  }
            }
		  }
		}
	  
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ContrastiveLossDenseLayer);
#endif

INSTANTIATE_CLASS(ContrastiveLossDenseLayer);
REGISTER_LAYER_CLASS(ContrastiveLossDense);

}  // namespace caffe
