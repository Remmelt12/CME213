#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/random.h>

// You may include other thrust headers if necessary.

#include "test_macros.h"

// You will need to call these functors from
// thrust functions in the code do not create new ones

// returns true if the char is not a lowercase letter
struct isnot_lowercase_alpha : thrust::unary_function<unsigned char, bool> {
   __host__ __device__ 
   bool operator()(unsigned char c){return(c<'a' || c>'z');} 
};

// convert an uppercase letter into a lowercase one
// do not use the builtin C function or anything from boost, etc.
struct upper_to_lower : thrust::unary_function<unsigned char, unsigned char> {

    __host__ __device__ 
    unsigned char operator()(unsigned char c)
    {
        if(check_(c)){
            return c+32;
        }
        else{
            return c;
        }
    }

    private:
    isnot_lowercase_alpha check_;
};

// apply a shift with appropriate wrapping
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
        else{
            return c+shift;
        }

    }

    apply_shift(thrust::device_ptr<unsigned int> begin,unsigned int period): 
        begin_(begin), period_(period) {}
    
    private:
    thrust::device_ptr<unsigned int> begin_;
    unsigned int period_;

};

// Returns a vector with the top 5 letter frequencies in text.
std::vector<double> getLetterFrequencyCpu(
    const std::vector<unsigned char>& text) {
    std::vector<unsigned int> freq(256);

    for(unsigned int i = 0; i < text.size(); ++i) {
        freq[tolower(text[i])]++;
    }

    unsigned int sum_chars = 0;

    for(unsigned char c = 'a'; c <= 'z'; ++c) {
        sum_chars += freq[c];
    }

    std::vector<double> freq_alpha_lower;

    for(unsigned char c = 'a'; c <= 'z'; ++c) {
        if(freq[c] > 0) {
            freq_alpha_lower.push_back(freq[c] / static_cast<double>(sum_chars));
        }
    }

    std::sort(freq_alpha_lower.begin(), freq_alpha_lower.end(),
              std::greater<double>());
    freq_alpha_lower.resize(min(static_cast<int>(freq_alpha_lower.size()), 5));

    return freq_alpha_lower;
}

// Print the top 5 letter frequencies and them.
std::vector<double> getLetterFrequencyGpu(
    const thrust::device_vector<unsigned char>& text) {

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

	for(int i =0; i<freq_alpha_lower.size();i++)
	{
		std::cout<< keys[i]<<": "<< freq_alpha_lower[i] <<std::endl; 	
	}


    return freq_alpha_lower;
}

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Didn't supply plain text and period!" << std::endl;
        return 1;
    }

    std::ifstream ifs(argv[1], std::ios::binary);

    if(!ifs.good()) {
        std::cerr << "Couldn't open text file!" << std::endl;
        return 1;
    }

    unsigned int period = atoi(argv[2]);

    if(period < 4) {
        std::cerr << "Period must be at least 4!" << std::endl;
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

    thrust::device_vector<unsigned char> text_clean;
    thrust::device_vector<unsigned char> device_text(text);
    // TODO: sanitize input to contain only a-z lowercase (use the
    // isnot_lowercase_alpha functor), calculate the number of characters
    // in the cleaned text and put the result in text_clean, make sure to
    // resize text_clean to the correct size!

    text_clean.resize(text.size());

    int numElements =-1;
    upper_to_lower cast= upper_to_lower();

    thrust::remove_copy_if(thrust::make_transform_iterator(device_text.begin(),cast),
                    thrust::make_transform_iterator(device_text.end(),cast),
                    text_clean.begin(),
                    isnot_lowercase_alpha());
    
    int non_alpha = thrust::count_if(text_clean.begin(),
                                     text_clean.end(),
                                    isnot_lowercase_alpha());


    numElements = text.size()-non_alpha; 
    text_clean.resize(numElements);


    std::cout << "\nBefore ciphering!" << std::endl << std::endl;
    std::vector<double> letterFreqGpu = getLetterFrequencyGpu(text_clean);
    std::vector<double> letterFreqCpu = getLetterFrequencyCpu(text);
    bool success = true;
    EXPECT_VECTOR_EQ_EPS(letterFreqCpu, letterFreqGpu, 1e-14, &success);
    PRINT_SUCCESS(success);

    thrust::device_vector<unsigned int> shifts(period);
    // TODO fill in shifts using thrust random number generation (make sure
    // not to allow 0-shifts, this would make for rather poor encryption).
    
    static thrust::default_random_engine rng(123);
    static thrust::uniform_int_distribution<int> gen(1,25);
    for (unsigned int i = 0; i < period; ++i) shifts[i] = gen(rng);


    std::cout << "\nEncryption key: ";

    for(int i = 0; i < period; ++i) {
        std::cout << static_cast<char>('a' + shifts[i]);
    }

    std::cout << std::endl;

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    // TODO: Apply the shifts to text_clean and place the result in
    // device_cipher_text.
    
    thrust::device_ptr<unsigned int> p = &(shifts[0]);
    
    apply_shift shift = apply_shift(p , period);

    thrust::transform(text_clean.begin(),
                      text_clean.end(),
                      thrust::make_counting_iterator(static_cast<int>(0)),
                      device_cipher_text.begin(),
                      shift);

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;

    std::cout << "After ciphering!" << std::endl << std::endl;
    getLetterFrequencyGpu(device_cipher_text);

    std::ofstream ofs("cipher_text.txt", std::ios::binary);

    ofs.write((char*)&host_cipher_text[0], numElements);

    ofs.close();

    return 0;
}
