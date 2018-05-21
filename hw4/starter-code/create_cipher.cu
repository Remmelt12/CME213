#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

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
    unsigned char operator()(unsigned char c, int position)
    {
        int key_pos=position % period_;
        char shift = begin_[key_pos]-97;
        return c+shift ; 
    }

    apply_shift(char* begin,unsigned int period): begin_(begin), period_(period) {}
    
    private:
        char* begin_;
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
    
    unsigned int sum_chars = 0;

    // WARNING: make sure you handle the case of not all letters appearing
    // in the text.


    // TODO calculate letter frequency
    // copy input data 
    thrust::device_vector<unsigned char> data(text);
    thrust::device_vector<unsigned int> freq;

    // sort data to bring equal elements together
    thrust::sort(data.begin(), data.end());

    // number of freq bins is equal to the maximum
    // value plus one
    unsigned int num_bins = data.back() + 1;

    // resize freq storage
    freq.resize(num_bins);
    freq_alpha_lower.resize(num_bins);

    // find the end of each bin of values
    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::upper_bound(data.begin(),data.end(),
                        search_begin,search_begin + num_bins,
                        freq.begin());

    // compute the freq by taking
    // differences of the cumulative
    // freq
    thrust::adjacent_difference(freq.begin(),
                                freq.end(),
                                freq.begin());

    // print the freq
    sum_chars=thrust::reduce(freq.begin(),freq.end());

    for(unsigned char c = 'a'; c <= 'z'; ++c) {
        if(freq[c] > 0) {
            freq_alpha_lower.push_back(freq[c] / static_cast<double>(sum_chars));
        }
    }

    /*
    thrust::transform(freq.begin(),freq.end(),freq_alpha_lower,[=]__host__
            __device__ (unsigned int i) double {return ((double) i)/((double)
                sum_chars) });
    */
    thrust::sort(freq_alpha_lower.begin(),freq_alpha_lower.end(),thrust::greater<double>());

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
    // TODO: sanitize input to contain only a-z lowercase (use the
    // isnot_lowercase_alpha functor), calculate the number of characters
    // in the cleaned text and put the result in text_clean, make sure to
    // resize text_clean to the correct size!


    int numElements =-1;
    numElements = text.size() - thrust::reduce(
			thrust::make_transform_iterator(text.begin(),isnot_lowercase_alpha()),
			thrust::make_transform_iterator(text.end(),isnot_lowercase_alpha()));

    text_clean.resize(numElements);

    upper_to_lower cast= upper_to_lower();
    thrust::copy_if(thrust::make_transform_iterator(text.begin(),cast),
                    thrust::make_transform_iterator(text.end(),cast),
                    isnot_lowercase_alpha());

    
    std::cout << "\nBefore ciphering!" << std::endl << std::endl;
    std::vector<double> letterFreqGpu = getLetterFrequencyGpu(text_clean);
    std::vector<double> letterFreqCpu = getLetterFrequencyCpu(text);
    bool success = true;
    EXPECT_VECTOR_EQ_EPS(letterFreqCpu, letterFreqGpu, 1e-14, &success);
    PRINT_SUCCESS(success);

    thrust::device_vector<unsigned int> shifts(period);
    // TODO fill in shifts using thrust random number generation (make sure
    // not to allow 0-shifts, this would make for rather poor encryption).


    std::cout << "\nEncryption key: ";

    for(int i = 0; i < period; ++i) {
        std::cout << static_cast<char>('a' + shifts[i]);
    }

    std::cout << std::endl;

    thrust::device_vector<unsigned char> device_cipher_text(numElements);

    // TODO: Apply the shifts to text_clean and place the result in
    // device_cipher_text.

    thrust::host_vector<unsigned char> host_cipher_text = device_cipher_text;

    std::cout << "After ciphering!" << std::endl << std::endl;
    getLetterFrequencyGpu(device_cipher_text);

    std::ofstream ofs("cipher_text.txt", std::ios::binary);

    ofs.write((char*)&host_cipher_text[0], numElements);

    ofs.close();

    return 0;
}
