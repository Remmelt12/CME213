#include <vector>
#include <fstream>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>

#include "strided_range_iterator.h"

// You will need to call these functors from thrust functions in the code
// do not create new ones

// this can be the same as in create_cipher.cu
struct apply_shift : thrust::binary_function<unsigned char, int,
        unsigned char> {

    __host__ __device__ 
    unsigned char operator()(unsigned char c, int position)
    {
        int key_pos=position % period_;
        unsigned char shift = begin_[key_pos];
        if(c+shift>'z'){
            return 'a'+((c+shift) % 123);
        }
		else if(c+shift<'a'){
            return 'z'+((c+shift) % 96);
		
		} 
        else{
            return c+shift;
        }
    }

    apply_shift(thrust::device_ptr<int> begin,unsigned int period): 
        begin_(begin), period_(period) {}
    
    private:
    thrust::device_ptr<int> begin_;
    unsigned int period_;

};

thrust::device_vector<unsigned char> getLetterFrequencyGpu(
    thrust::device_vector<unsigned char>& text) {

    std::vector<double> freq_alpha_lower;
    
    unsigned int sum_chars = text.size();

    // WARNING: make sure you handle the case of not all letters appearing
    // in the text.

    // calculate letter frequency
    // copy input data 
    thrust::device_vector<unsigned char> data=text;

    // sort data to bring equal elements together
    thrust::sort(data.begin(), data.end());

    // number of freq bins is equal to the maximum
    // value plus one
    unsigned int num_bins = thrust::inner_product(thrust::device,data.begin(), data.end() - 1,
                                                data.begin() + 1,
                                                1,
                                                thrust::plus<int>(),
                                                thrust::not_equal_to<int>());

    thrust::device_vector<unsigned char> keys(num_bins);
    thrust::device_vector<double> counts(num_bins);
    thrust::constant_iterator<double> size_inv(1.0/(double) sum_chars );

    thrust::reduce_by_key(thrust::device,data.begin(), data.end(),
                          size_inv,
                          keys.begin(),
                          counts.begin());

    thrust::sort_by_key(thrust::device,counts.begin(),
            counts.end(),keys.begin(),thrust::greater<double>());

    freq_alpha_lower.resize(num_bins);
    thrust::copy(counts.begin(),counts.end(),freq_alpha_lower.begin());

    freq_alpha_lower.resize(min(static_cast<int>(freq_alpha_lower.size()), 5));
	keys.resize(min(static_cast<int>(keys.size()), 5));

    return keys;
}


int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "No cipher text given!" << std::endl;
        return 1;
    }

    // First load the text
    std::ifstream ifs(argv[1], std::ios::binary);

    if(!ifs.good()) {
        std::cerr << "Couldn't open book file!" << std::endl;
        return 1;
    }

    // load the file into text
    std::vector<unsigned char> text;

    ifs.seekg(0, std::ios::end); // seek to end of file
    int length = ifs.tellg();    // get distance from beginning
    ifs.seekg(0, std::ios::beg); // move back to beginning

    text.resize(length);
    ifs.read((char*)&text[0], length);

    ifs.close();

    // we assume the cipher text has been sanitized
    thrust::device_vector<unsigned char> text_clean = text;

    // now we crack the Vigenere cipher
    // first we need to determine the key length
    // use the kappa index of coincidence
    int keyLength = 0;
    {
        bool found = false;
        int shift_idx = 4; // Start at index 4.

        while(!found) {
            // TODO: Use thrust to compute the number of characters that match
            // when shifting text_clean by shift_idx.

            int numMatches =thrust::inner_product(thrust::device,
                                    text_clean.begin(),
                                    text_clean.end() -shift_idx,
                                    text_clean.begin() + shift_idx,
                                    1,
                                    thrust::plus<int>(),
                                    thrust::equal_to<int>());

            double ioc = numMatches /
                         static_cast<double>((text_clean.size() - shift_idx) / 26.);

            std::cout << "Period " << shift_idx << " ioc: " << ioc << std::endl;
            if(ioc > 1.6) {
                if(keyLength == 0) {
                    keyLength = shift_idx;
                    shift_idx = 2 * shift_idx - 1; // check double the period to make sure
                } else if(2 * keyLength == shift_idx) {
                    found = true;
                } else {
                    std::cout << "Unusual pattern in text!" << std::endl;
                    exit(1);
                }
            }

            ++shift_idx;
        }
    }

    std::cout << "keyLength: " << keyLength << std::endl;

    // once we know the key length, then we can do frequency analysis on each
    // pos mod length allowing us to easily break each cipher independently
    // you will find the strided_range useful
    // it is located in strided_range_iterator.h and an example
    // of how to use it is located in the that file
    thrust::device_vector<unsigned char> text_copy = text_clean;
    thrust::device_vector<int> dShifts(keyLength);
    typedef thrust::device_vector<unsigned char>::iterator Iterator;


    // TODO: Now that you have determined the length of the key, you need to
    // compute the actual key. To do so, perform keyLength individual frequency
    // analyses on text_copy to find the shift which aligns the most common
    // character in text_copy with the character 'e'. Fill up the
    // dShifts vector with the correct shifts.

	for(int i=0;i<keyLength;i++)
	{
		strided_range<Iterator> stride_it(text_copy.begin()+i, text_copy.end(),
				keyLength);

	
		thrust::device_vector<unsigned char>
			sub_text((text_copy.size()+keyLength-1)/keyLength);

		thrust::copy(stride_it.begin(),
					stride_it.end(),
					sub_text.begin()); 	
		thrust::device_vector<unsigned char>
			freq=getLetterFrequencyGpu(stride_it);


		dShifts[i]= 'e'-freq[0] ;
	
	}

    std::cout << "\nEncryption key: ";

    for(unsigned int i = 0; i < keyLength; ++i)
        std::cout << static_cast<char>('a' - (dShifts[i] <= 0 ? dShifts[i] :
                                              dShifts[i] - 26));

    std::cout << std::endl;

    // take the shifts and transform cipher text back to plain text
    // TODO : transform the cipher text back to the plain text by using the
    // apply_shift functor.
	
    thrust::device_ptr<int> p = &(dShifts[0]);
    
    apply_shift shift = apply_shift(p , keyLength);

    thrust::transform(text_copy.begin(),
                      text_copy.end(),
                      thrust::make_counting_iterator(static_cast<int>(0)),
                      text_clean.begin(),
                      shift);



    thrust::host_vector<unsigned char> h_plain_text = text_clean;

    std::ofstream ofs("plain_text.txt", std::ios::binary);
    ofs.write((char*)&h_plain_text[0], h_plain_text.size());
    ofs.close();

    return 0;
}
