# AUNet
Automatic, recursive-style generation of a UNet for an arbitrary, divisible by 4 input image size

The automatic Unet generation was implemented in order to alleviate hyperparameter tuning.
As opposed to many other Unet implementations, this class only needs the primefactor two twice
in its prime factorization of the image dimension: 300x300 works, 304x304 works, 296x296 works
and so on. The number of Down/Up convolutions can be chosen using the "depth" parameter. The
number of channels in the top convolution can be chosen via "top_channels". The channels are
halfed / doubled with every Down/Up convolution block. With "out_channels", the number of classes
to be segmented can be adapted.


## Background information

Canonical U-Nets do not work with arbitrary input
sizes due to the 2x2 stride 2 maxpooling layers when
encoding. Assuming valid padding, the problem arises if
the image dimension(s) has less numbers 2 in their prime
factorization than the depth of the U, i. e. the number
of convolutional blocks in the encoder. Consider, for
example, an image of size 300x300 and a depth 3
of the U. Than, the processing of the image dimensions
during encoding looks like:

300x300 => 150x150 => 75x75 => 37x37

In the last step, 75 is not divisible by 2, so the next
size is round(75/2) = 37. This leads to problems while
decoding. The first decoding block does:

37x37 => 74x74

Now, we face the problem that we cannot concatenate the
encoding 75x75 activations with the 74x74 activations
without padding/cropping or similar.

The problem is solved in the following manner. While
encoding, we crop if the next block would produce
an odd dimension: 300x300 looks at 75x75 and crops
its ouput from 150x150 symmetrically to 148x148.

This way we get for the encoding:

300x300 => 148x148 => 72x72 => 36x36

Now, we are able to pad symmetrical when
decoding, as all encoding dimensions are even.

Note that we need the prime factor two twice
in the image dimensions since we symmetrically crop
on the output.

