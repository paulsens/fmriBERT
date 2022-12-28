# "Final" pretraining results
BOTH_averages={"first_epoch":{'t_loss': 0.06313666666666666, 'tb_acc': 71.44450000000002, 'tm_acc': 0.9939999999999999, 't_correct': [3116.5, 4021.4166666666665], 'tb_loss': 0.5834583333333333, 'tm_loss': 0.0052, 'v_loss': 0.056225, 'vb_acc': 75.327, 'vm_acc': 0.9999166666666666, 'vb_loss': 0.5519333333333334, 'vm_loss': 0.0011333333333333332, 'v_correct': [247.66666666666666, 346.0833333333333]},

             "fifth_epoch":{'t_loss': 0.04649333333333334, 'tb_acc': 85.64274999999999, 'tm_acc': 0.9990000000000001, 't_correct': [3994.25, 4562.166666666667], 'tb_loss': 0.4563083333333333, 'tm_loss': 0.0013333333333333333, 'v_loss': 0.049125, 'vb_acc': 82.23816666666666, 'vm_acc': 0.9998333333333335, 'vb_loss': 0.4841833333333334, 'vm_loss': 0.000775, 'v_correct': [278.0833333333333, 379.1666666666667]},
               }

NTPe4_averages={"first_epoch":{'tb_loss': 0.6290681818181817, 'tb_acc': 66.82227272727272, 't_correct': [2915.818181818182, 3759.7272727272725], 'vb_loss': 0.5932381818181818, 'vb_acc': 70.54554545454545, 'v_correct': [223.45454545454547, 340.90909090909093]},

    "fifth_epoch":{'tb_loss': 0.5719327272727273, 'tb_acc': 73.39963636363636, 't_correct': [3075.5454545454545, 4257.090909090909], 'vb_loss': 0.5549427272727272, 'vb_acc': 75.375, 'v_correct': [238.0909090909091, 364.1818181818182]}
                }

NTPe5_averages={"first_epoch":{'tb_loss': 0.5780108333333334, 'tb_acc': 71.71158333333332, 't_correct': [2935.9166666666665, 4228.666666666667], 'vb_loss': 0.5331199999999999, 'vb_acc': 76.82833333333333, 'v_correct': [247.91666666666666, 366.0833333333333]},

    "fifth_epoch":{'tb_loss': 0.47513999999999995, 'tb_acc': 83.40633333333334, 't_correct': [3736.1666666666665, 4596.833333333333], 'vb_loss': 0.5251208333333334, 'vb_acc': 77.97525, 'v_correct': [288.75, 334.4166666666667]}
                }

MBM_averages={"first_epoch":{'tm_loss': 0.0046725, 'tm_acc': 0.9949999999999998, 't_correct': [682.8333333333334, 4324.416666666667], 'vm_loss': 0.0011933333333333334, 'vm_acc': 1.0, 'v_correct': [5.083333333333333, 393.0]},

    "fifth_epoch":{'tm_loss': 0.0016575000000000001, 'tm_acc': 0.9983333333333332, 't_correct': [302.25, 4562.666666666667], 'vm_loss': 0.0007066666666666667, 'vm_acc': 0.9996666666666667, 'v_correct': [4.5, 387.9166666666667]}
              }

#SORTED
# currently these lists are all for fifth epoch only
BOTH_sorted={
    "tb_acc":[(83.934, 1), (84.014, 10), (84.935, 6), (84.975, 0), (85.005, 5), (85.14, 11), (85.696, 4), (85.886, 2), (85.966, 7), (85.976, 9), (86.376, 8), (89.81, 3)],
    "vb_acc":[(78.608, 11), (78.875, 6), (79.75, 1), (79.875, 9), (80.375, 0), (80.75, 10), (81.5, 4), (81.625, 7), (85.0, 5), (85.875, 2), (87.25, 3), (87.375, 8)],
    "tb_loss":[(0.4119, 3), (0.4456, 11), (0.4462, 8), (0.4486, 9), (0.45, 2), (0.451, 4), (0.4582, 0), (0.4587, 5), (0.461, 6), (0.4684, 1), (0.4694, 10), (0.5067, 7)],
    "vb_loss":[(0.436, 8), (0.441, 3), (0.4505, 2), (0.4618, 5), (0.4854, 11), (0.4919, 4), (0.4963, 7), (0.5028, 10), (0.503, 1), (0.5071, 0), (0.5116, 9), (0.5228, 6)],
    "tm_loss":[(0.0012, 6), (0.0012, 11), (0.0013, 1), (0.0013, 2), (0.0013, 4), (0.0013, 9), (0.0014, 0), (0.0014, 3), (0.0014, 5), (0.0014, 7), (0.0014, 8), (0.0014, 10)],
    "vm_loss":[(0.0003, 9), (0.0004, 10), (0.0005, 8), (0.0006, 2), (0.0006, 6), (0.0006, 7), (0.0007, 5), (0.0008, 3), (0.0009, 0), (0.0009, 11), (0.0012, 4), (0.0018, 1)],
    "t_loss":[(0.04247, 3), (0.04592, 8), (0.04607, 9), (0.04616, 2), (0.04625, 4), (0.04634, 7), (0.04707, 0), (0.04709, 5), (0.04713, 11), (0.0472, 6), (0.04799, 1), (0.04823, 10)],
    "v_loss":[(0.0441, 8), (0.0448, 3), (0.0456, 2), (0.0468, 5), (0.0493, 11), (0.0502, 7), (0.0503, 4), (0.0506, 10), (0.0515, 9), (0.0516, 0), (0.0519, 1), (0.0528, 6)],
}
BOTH_sorted_first={"vb_acc":[(68.125, 4), (71.0, 5), (73.0, 7), (73.924, 11), (74.125, 10), (74.875, 3), (77.25, 8), (77.25, 9), (77.625, 0), (78.25, 2), (79.0, 6), (79.5, 1)]
}

NTPe5_sorted={"tb_acc":[(82.783, 10), (83.003, 0), (83.013, 9), (83.243, 5), (83.263, 4), (83.373, 8), (83.393, 3), (83.463, 7), (83.744, 6), (83.854, 2), (83.86, 11), (83.884, 1)],
            "vb_acc":[(73.875, 4), (75.125, 8), (76.203, 11), (76.25, 6), (76.625, 0), (77.125, 9), (78.25, 5), (79.125, 2), (79.75, 1), (80.25, 10), (80.875, 7), (82.25, 3)],
            "tb_loss":[(0.47016, 1), (0.47123, 6), (0.47143, 2), (0.47173, 11), (0.4736, 7), (0.47439, 3), (0.47682, 5), (0.47707, 8), (0.47806, 4), (0.47828, 0), (0.47929, 9), (0.47962, 10)],
            "vb_loss":[(0.48564, 3), (0.50074, 7), (0.5052, 1), (0.5072, 10), (0.52075, 5), (0.52151, 2), (0.53457, 9), (0.53814, 0), (0.53867, 11), (0.54041, 6), (0.54272, 8), (0.5659, 4)]
}
NTPe5_sorted_first={"vb_acc":[(72.125, 8), (72.125, 9), (75.19, 11), (75.625, 2), (76.625, 0), (77.375, 10), (78.125, 1), (78.25, 6), (78.75, 3), (78.875, 4), (79.125, 5), (79.75, 7)]
}

MBM_sorted={"tm_loss":[(0.00151, 5), (0.00159, 1), (0.00165, 0), (0.00166, 9), (0.00167, 2), (0.00167, 6), (0.00167, 10), (0.00168, 7), (0.00168, 11), (0.00169, 8), (0.0017, 4), (0.00172, 3)],
            "vm_loss":[(0.00028, 9), (0.00029, 6), (0.00031, 10), (0.00033, 2), (0.00049, 0), (0.00053, 5), (0.00054, 7), (0.00055, 8), (0.00079, 3), (0.0009, 11), (0.00171, 4), (0.00176, 1)]
}


# t = training, b = ntp task, m = mbm task, correct is two numbers separated by a comma, number of negative,positive inputs labelled correctly

#order:
# tloss tbacc tmacc tcorrect tbloss tmloss vloss vbacc vmacc vbloss vmloss vcorrect
BOTH_keys=["t_loss", "tb_acc", "tm_acc", "t_correct", "tb_loss", "tm_loss", "v_loss", "vb_acc", "vm_acc", "vb_loss", "vm_loss", "v_correct"]
BOTH={
"first_epoch":[".06327 71.421 .994 2989,4146 .5864 .0051 .0545 77.625 1.0 .5273 .0019 307,314",
               ".06215 72.442 .994 3064,4173 .5750 .0052 .0525 79.5 1.0 .5140 .0012 274,362",
               ".06437 70.04 .994 3039,3958 .5963 .0053 .0531 78.250 1.0 .5270 .0004 228,398",
               ".06398 70.651 .994 3022,4036 .5934 .0052 .0572 74.875 1.0 .5562 .0018 270,329",
               ".06296 71.872 .994 3085,4095 .5830 .0052 .0626 68.125 1.0 .6214 .0006 203,243",
               ".06268 72.052 .994 3093,4105 .5790 .0053 .0598 71 .999 .5875 .0011 288,280",
               ".06187 72.653 .994 3270,3988 .5718 .0052 .0529 79 1.0 .5228 .0007 257,375",
               ".06322 71.301 .994 3219,3904 .5857 .0052 .0565 73 1.0 .5586 .0007 286,298",
               ".06300 71.361 .994 3062,4067 .5834 .0052 .0547 77.25 1.0 .5329 .0016 222,396",
               ".06336 71.231 .994 3190,3926 .5866 .0052 .0553 77.25 1.0 .5388 .0015 228,390",
               ".06434 70.1 .994 3224,3779 .5973 .0051 .0567 74.125 1.0 .5632 .0004 201,392",
               ".06244 72.21 .994 3141,4080 .5636 .0052 .0589 73.924 1.0 .5735 .0017 208,376",
             ],

"fifth_epoch":[".04707 84.975 .999 3944,4545 .4582 .0014 .0516 80.375 1.0 .5071 .0009 246,397",
               ".04799 83.934 .999 3890,4495 .4684 .0013 .05190 79.75 .999 .5030 .0018 317,321",
               ".04616 85.886 .999 4045,4535 .4500 .0013 .0456 85.875 1.0 .4505 .0006 293,394",
               ".04247 89.810 .999 4353,4619 .4119 .0014 .0448 87.250 1.0 .4410 .0008 303,395",
               ".04625 85.696 .999 4059,4502 .4510 .0013 .0503 81.5 .999 .4919 .0012 257,395",
               ".04709 85.005 .999 3929,4563 .4587 .0014 .0468 85 1.0 .4618 .0007 299,381",
               ".0472 84.935 .999 3869,4616 .4610 .0012 .0528 78.875 1.0 .5228 .0006 267,364",
               ".04634 85.966 .999 4028,4560 .5067 .0014 .0502 81.625 1.0 .4963 .0006 296,357",
               ".04592 86.376 .999 4025,4604 .4462 .0014 .0441 87.375 1.0 .436 .0005 321,378",
               ".04607 85.976 .999 3955,4634 .4486 .0013 .0515 79.875 1.0 .5116 .0003 241,398",
               ".04823 84.014 .999 3850,4543 .4694 .0014 .0506 80.75 1.0 .5028 .0004 264,382",
               ".04713 85.14 .999 3984,4530 .4456 .0012 .0493 78.608 1.0 .4854 .0009 233,388",
            ]
}


# order is tbloss tbacc correct vbloss vbacc vcorrect
NTP_keys=["tb_loss", "tb_acc", "t_correct", "vb_loss", "vb_acc", "v_correct"]
# the binaryonly runs with LR set to e-4 (default)
NTPe4 = {
"first_epoch":[".61511 68.198 2902,3911 .55558 74.625 232,365",
               ".61535 68.398 2876,3957 .57372 73.25 199,387",
               ".63676 65.415 2593,3942 .62338 67.125 183,354",
               ".63966 65.576 3044,3507 .68207 58.375 256,211",
               ".63181 66.617 2912,3743 .54878 76.126 249,360",
               ".64875 64.845 3150,3328 .53884 77.5 240,380",
               ".63346 66.707 3000,3664 .59772 70.625 189,376",
               ".60121 70.15 2849,4159 .52958 78.25 239,387",
               ".63134 66.577 2896,3755 .62645 67.375 193,346",
               ".64087 65.295 3030,3493 .61698 68 243,301",
               ".62543 67.267 2822,3898 .63252 64.75 235,283",
               ],

"fifth_epoch":[".56157 74.665 3065,4394 .52809 78.500 234,394",
               ".5832 71.77 3185,3985 .52841 77.375 234,385",
               ".59570 70.781 2721,4350 .54743 76.5 245,367",
               ".56479 73.834 3063,4313 .56674 75.25 269,325",
               ".55086 75.435 3265,4271 .53379 77.375 264,355",
               ".54753 76.016 3150,4444 .54942 75.75 240,366",
               ".57580 73.203 3038,4275 .54656 76.375 230,381",
               ".59759 70.771 3017,4053 .56040 74.75 236,362",
               ".56759 74.304 3105,4318 .58966 71.625 227,346",
               ".56750 73.794 3195,4177 .57419 73.25 210,376",
               ".57913 72.823 3027,4248 .57968 72.375 230,349",
               ]
}

# binaryonly run with LR set to e-5 to try to account for the 0.1 factor applied to binary_loss in BOTH training
NTPe5={
    "first_epoch":[".57452 72.052 2938,4260 .53564 76.625 264,349",
                   ".57657 72.162 2915,4294 .52171 78.125 236,389",
                   ".57799 71.562 2907,4242 .53667 75.625 261,344",
                   ".58098 71.201 2927,4186 .52837 78.750 281,349",
                   ".57682 71.982 2949,4242 .52153 78.875 237,394",
                   ".57919 71.582 2958,4193 .51718 79.125 234,399",
                   ".57482 72.132 2928,4278 .52445 78.25 239,387",
                   ".57837 71.762 2951,4218 .51394 79.75 242,396",
                   ".58103 71.291 2893,4229 .55585 72.125 281,296",
                   ".58089 71.131 2906,4200 .56013 72.125 243,334",
                   ".57644 71.942 2930,4257 .53290 77.375 239,380",
                   ".57851 71.74 3029,4145 .54907 75.19 218,376",
    ],

    "fifth_epoch":[".47828 83.003 3703,4589 .53814 76.625 301,312",
            ".47016 83.884 3801,4579 .50520 79.75 308,330",
            ".47143 83.854 3726,4651 .52151 79.125 314,319",
        ".47439 83.393 3751,4580 .48564 82.250 302,356",
        ".47806 83.263 3688,4630 .56590 73.875 263,328",
        ".47682 83.243 3738,4578 .52075 78.250 265,361",
        ".47123 83.744 3755,4611 .54041 76.25 257,353",
        ".47360 83.463 3763,4575 .50074 80.875 269,378",
        ".47707 83.373 3698,4631 .54272 75.125 335,266",
        ".47929 83.013 3693,4600 .53457 77.125 280,337",
        ".47962 82.783 3748,4522 .50720 80.250 298,344",
        ".47173 83.86 3770,4616 .53867 76.203 273,329"
    ]
}


# order:
# tmloss tmacc tcorrect vmloss vmacc vcorrect
MBM_keys = ["tm_loss", "tm_acc", "t_correct", "vm_loss", "vm_acc", "v_correct"]
MBM={ "first_epoch":[".00466 .995 252,4735 .00246 1.0 0,400",
                     ".00471 .995 830,4173 .00161 1.0 0,400",
                     ".00473 .995 1391,3627 .00092 1.0 0,400",
                     ".00471 .995 1053,3956 .00133 1.0 0,400",
                     ".00467 .995 246,4746 .00079 1.0 0,400",
                     ".00464 .995 376,4644 .00059 1.0 0,400",
                     ".00464 .995 750,4265 .00117 1.0 0,400",
                     ".00466 .995 1048,3933 .0006 1.0 61,321",
                     ".00460 .995 429,4554 .00226 1.0 0,400",
                     ".00469 .995 498,4533 .00107 1.0 0,400",
                     ".00466 .995 1017,4044 .00089 1.0 0,400",
                     ".00470 .995 304,4683 .00063 1.0 0,395",
],
    "fifth_epoch":[".00165 .999 125,4768 .00049 1.0 0,400",
                   ".00159 .999 351,4512 .00176 .999 0,400",
                   ".00167 .998 509,4275 .00033 1.0 0,400",
                   ".00172 .998 214,4691 .00079 .999 0,400",
                   ".0017 .998 120,4803 .00171 .999 0,400",
                  ".00151 .999 141,4807 .00053 1.0 0,400",
                   ".00167 .998 478,4381 .00029 1.0 0,400",
                   ".00168 .998 529,4240 .00054 1.0 22,330",
                   ".00169 .998 348,4469 .00055 .999 1,400",
                   ".00166 .999 186,4712 .00028 1.0 0,400",
                   ".00167 .998 281,4615 .00031 1.0 31,330",
                   ".00168 .998 345,4479 .0009 1.0 0,395"
    ]
}

#BASELINES
BOTH_baseline={"first_epoch":[".50634 48.909 .004 2879,2007 .6979 .4851 .5043 46.75 -.006 .6948 .4831 288,86"],
            "fifth_epoch":[".50670 49.459 .004 2900,2041 .6977 .4855 .5043 46.750 -.006 .6948 .4831 288,86"]
}

NTP_baseline={"first_epoch":[".69794 48.909 2879,2007 .69479 46.750 288,86"],
              "fifth_epoch":[".69765 49.459 2900,2041 .69479 46.750 288,86"]
              }

MBM_baseline={
    "first_epoch":[".48506 .004 2879,2007 .48313 -.006 288,86"],
    "fifth_epoch":[".48548 .004 2900,2041 .48313 -.006 288,86"]
}



# "official" pretraining results, broken self attention mechanism
BOTHold_keys = ["train_ntpacc", "train_mbmacc", "train_correct", "train_ntploss", "train_mbmloss", "val_ntpacc", "val_mbmacc", "val_ntploss", "val_mbmloss", "val_correct"]
BOTHold={
"fifth_epoch":[
"78.629 .999 3346,4509 .3133 .0003 57.75 1.0 .542 .0006 306,156",
"77.497 .999 3276,4466 .3133 .0002 79 1.0 .3142 .0009 236,396",
"76.106 .999 3256,4347 .3137 .00005 64.5 1.0 .3485 .0004 277,239",
"76.747 .999 3264,4403, .3133 .0005 79.25 1.0 .3135 .0003 252,382",
"78.929 .999 3382,4503 .3133 .0009 64.625 .999 1.313 .0011 285,232",
"79.109 .999 3437,4466 .3133 .0063 79.125 1.0 1.3133 .0008 265, 368",
"77.878 .999 3358,4422 .3133 .0033 63.75 1.0 .3133 .0012 291,219",
"77.898 .999 3381,4401 1.3132 .0002 71.375 1.0 1.3131 .0003 273,298",
"78.218 .999 3380,4434 1.3132 .0003 60.125 1.0 .9973 .0007 315,166",
"78.709 .999 3329,4534 1.3124 .0005 57.25 1.0 .315 .00004 273,185",
"78.288 .999 3410,4411 1.3132 .0006 49.75 1.0 .3133 .00004 336,62",
"77.64 .999 3342,4422 .3166 .0007 45.57 1.0 1.1766 .0004 337,23"
]
    ,
"first_epoch":[
"61.251 .994 2850,3269 .3328 .0003 68.5 1.0 .4831 .0025 232,316",
"58.959 .994 2930,2960 .3362 .0011 72 .999 .5782 .0009 261,315",
"57.037 .994 2843,2855 .3595 .0002 61.75 1.0 .4422 .0002 335,159",
"58.959 .994 2904,2986 .3210 .003 73.5 1.0 .4392 .0011 260,328",
"60.951 .994 2802,3287 .3204 .0001 69.25 1.0 1.1957 .0008 249,305",
"60.981 .994 2819,3273 .3404 .0006 73 1.0 .9334 .0002 269,315",
"61.792 .994 2824,3349 .3187 .0006 68.5 1.0 .3210 .0004 222,326",
"61.081 .994 2904,3198 .3222 .0009 72.375 1.0 1.0258 .0004 283,296",
"61.341 .994 2888,3240 .3230 .0002 62.5 .999 1.0505 .0029 285,215",
"61.972 .994 2858,3333 .3260 .001 54.625 1.0 .5592 .0028 320,117",
"62.402 .994 2859,3375 .3172 .0012 49.375 .999 .3142 .0002 361,34",
"63.410 .994 3091,3250 .3165 .0024 49.114 1.0 .5446 .0003 285,103",
]
}

NTPold_keys = ["train_ntploss", "train_ntpacc", "train_correct", "val_nptloss", "val_ntpacc", "val_correct"]
NTPold={
    "fifth_epoch":[
    ".53212 77.758 3151,4617 .61431 69.5 284,272",
    ".53057 77.868 3243,4536 .60844 70 231,329",
    ".55187 75.916 3302,4282 .69382 61.375 247,244",
    ".51626 79.319 3285,4639 .53888 75.625 273,332",
    ".53456 77.417 3297,4437 .62323 68.875 309,242",
    ".51576 79.239 3323,4593 .51759 79.125 263,370",
    ".52483 78.358 3290,4538 .52357 78.625 252,377",
    ".52194 78.609 3319,4534 .59474 70.75 290,276",
    ".53162 77.888 3349,4432 .64797 66 287,241",
    ".52221 78.789 3316,4555 .75250 54.750 303,135",
    ".52617 78.539 3252,4594 .83724 47 291,85",
    ".54774 76.09 3312,4297 .88553 42.405 332,3",
    ]
,
"first_epoch":[
".62827 64.575 2804,3647 .58668 72.25 237,341",
".6358 63.994 2871,3522 .60061 70.5 215,349",
".63549 64.585 3001,3451 .60425 70.125 200,361",
".64843 62.763 2802,3468 .57945 72.5 249,331",
".63882 63.644 2686,3672 .71764 54.25 267,167",
".64471 63.083 2710,3592 .61998 64.125 318,195",
".6404 63.453 2768,3571 .57255 73.375 276,311",
".64079 63.393 2728,3605 .65786 63 331,171",
".63631 64.464 2766,3674 .65061 62.5 202,298",
".63179 64.755 2850,3619 .73930 56.875 345,110",
".63706 63.934 2806,3581 .83551 46.625 327,46",
".62799 66.03 3233,3370 .80678 49.62 214,178"
]
}

MBMold_keys = ["train_mbmloss", "train_mbmacc", "val_mbmloss", "val_mbmacc"]
MBMold={
"fifth_epoch":[
".00166 .998 .00038 1.0",
".00151 .999 .00099 1.0",
".00159 .999 .00035 1.0",
".00168 .998 .00065 1.0",
".00167 .998 .00086 1.0",
".00152 .999 .00036 1.0",
".00159 .999 .0003 1.0",
".00164 .999 .00076 .999",
".00161 .999 .00043 1.0",
".00160 .999 .00028 1.0",
".00169 .998 .00032 1.0",
".00157 .999 .0007 1.0"
    ],

"first_epoch":[
".00465 .995 .0013 1.0",
".00467 .995 .00243 1.0",
".00468 .995 .00074 1.0",
".00470 .995 .00109 1.0",
".00470 .995 .00086 1.0",
".00467 .995 .00044 1.0",
".00469 .995 .00113 1.0",
".00467 .995 .00083 1.0",
".00466 .995 .00051 1.0",
".00471 .994 .00165 1.0",
".00468 .995 .00052 1.0",
".00470 .994 .0008 1.0"
    ]
}