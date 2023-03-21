# This source code is from https://github.com/usuyama/ePillID-benchmark/blob/master/src/image_augmentators.py
#
# A copy of the license is provided below
#
# The MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)


def get_imgaug_sequences(low_gblur = 1.0, 
    high_gblur = 3.0, addgn_base_ref = 0.01, 
    addgn_base_cons = 0.001, rot_angle = 180, 
    max_scale = 1.0, add_perspective = False
):
    affine_seq = iaa.Sequential([
            iaa.Affine(
                rotate=(-rot_angle, rot_angle),
                scale=(0.8, max_scale),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            ),
            sometimes(iaa.Affine(
                shear=(-4, 4),
            ))
        ])
    
    affine_list = [affine_seq]

    contrast_list = [
            iaa.Sequential([
                iaa.LinearContrast((0.7, 1.0), per_channel=False), # change contrast
                iaa.Add((-30, 30), per_channel=False), # change brightness
            ]),
            iaa.Sequential([
                iaa.LinearContrast((0.4, 1.0), per_channel=False), # change contrast
                iaa.Add((-80, 80), per_channel=False), # change brightness
            ])            
        ]

    if add_perspective:
        print("Adding perspective transform to augmentation")
        affine_list =  affine_list + [
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ]
                          
            

        contrast_list = contrast_list + [ 
            iaa.GammaContrast((0.5, 1.7), per_channel=True),
            iaa.SigmoidContrast(gain=(8, 12), cutoff=(0.2,0.8), per_channel=False)
             ]

    

    ref_seq = iaa.Sequential(affine_list + [
        iaa.OneOf(contrast_list),
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 3*addgn_base_ref*255), per_channel=0.5),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, addgn_base_ref*255), per_channel=0.5),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, high_gblur)),
            iaa.GaussianBlur(sigma=(0, low_gblur)),
        ])
    ])

    cons_seq = iaa.Sequential(affine_list + [
        iaa.LinearContrast((0.9, 1.1), per_channel=False),
        iaa.Add((-10, 10), per_channel=False),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 5*addgn_base_cons*255), per_channel=0.5),
        iaa.GaussianBlur(sigma=(0, low_gblur)),
    ])
    
    return affine_seq, ref_seq, cons_seq