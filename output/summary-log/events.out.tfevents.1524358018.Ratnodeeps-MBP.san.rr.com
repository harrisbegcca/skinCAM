       ЃK"	  ріЖжAbrain.Event:2ещзYiЙ     ^Yi	эріЖжA"мђ

R
Placeholder_xPlaceholder*
dtype0*
_output_shapes
:*
shape:
R
Placeholder_yPlaceholder*
_output_shapes
:*
shape:*
dtype0
k
Conv_Reshape/shapeConst*
_output_shapes
:*%
valueB"џџџџd   d      *
dtype0

Conv_ReshapeReshapePlaceholder_xConv_Reshape/shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџdd
g
Kernel_0/shapeConst*%
valueB"
   
      
   *
dtype0*
_output_shapes
:
R
Kernel_0/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_0/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_0/RandomStandardNormalRandomStandardNormalKernel_0/shape*

seed *
T0*
dtype0*&
_output_shapes
:


*
seed2 
t
Kernel_0/mulMulKernel_0/RandomStandardNormalKernel_0/stddev*
T0*&
_output_shapes
:



]
Kernel_0AddKernel_0/mulKernel_0/mean*&
_output_shapes
:


*
T0

Variable
VariableV2*
dtype0*&
_output_shapes
:


*
	container *
shape:


*
shared_name 
Є
Variable/AssignAssignVariableKernel_0*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:



q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:



з
conv2d_0Conv2DConv_ReshapeVariable/read*/
_output_shapes
:џџџџџџџџџdd
*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
l
Kernel_Bias_0/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   d   d   
   
W
Kernel_Bias_0/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_0/stddevConst*
_output_shapes
: *
valueB
 *ЭЬL=*
dtype0
І
"Kernel_Bias_0/RandomStandardNormalRandomStandardNormalKernel_Bias_0/shape*
T0*
dtype0*&
_output_shapes
:dd
*
seed2 *

seed 

Kernel_Bias_0/mulMul"Kernel_Bias_0/RandomStandardNormalKernel_Bias_0/stddev*
T0*&
_output_shapes
:dd

l
Kernel_Bias_0AddKernel_Bias_0/mulKernel_Bias_0/mean*
T0*&
_output_shapes
:dd



Variable_1
VariableV2*
shared_name *
dtype0*&
_output_shapes
:dd
*
	container *
shape:dd

Џ
Variable_1/AssignAssign
Variable_1Kernel_Bias_0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:dd
*
use_locking(*
T0
w
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*&
_output_shapes
:dd

_
addAddconv2d_0Variable_1/read*
T0*/
_output_shapes
:џџџџџџџџџdd

M
relu_0Reluadd*
T0*/
_output_shapes
:џџџџџџџџџdd

Ѕ
Pool_0MaxPoolrelu_0*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22

g
Kernel_1/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_1/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_1/RandomStandardNormalRandomStandardNormalKernel_1/shape*&
_output_shapes
:



*
seed2 *

seed *
T0*
dtype0
t
Kernel_1/mulMulKernel_1/RandomStandardNormalKernel_1/stddev*&
_output_shapes
:



*
T0
]
Kernel_1AddKernel_1/mulKernel_1/mean*
T0*&
_output_shapes
:






Variable_2
VariableV2*
shared_name *
dtype0*&
_output_shapes
:



*
	container *
shape:




Њ
Variable_2/AssignAssign
Variable_2Kernel_1*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:




w
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*&
_output_shapes
:




г
conv2d_1Conv2DPool_0Variable_2/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22
*
	dilations

l
Kernel_Bias_1/shapeConst*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_1/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_1/RandomStandardNormalRandomStandardNormalKernel_Bias_1/shape*

seed *
T0*
dtype0*&
_output_shapes
:22
*
seed2 

Kernel_Bias_1/mulMul"Kernel_Bias_1/RandomStandardNormalKernel_Bias_1/stddev*
T0*&
_output_shapes
:22

l
Kernel_Bias_1AddKernel_Bias_1/mulKernel_Bias_1/mean*
T0*&
_output_shapes
:22



Variable_3
VariableV2*
dtype0*&
_output_shapes
:22
*
	container *
shape:22
*
shared_name 
Џ
Variable_3/AssignAssign
Variable_3Kernel_Bias_1*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:22

w
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*&
_output_shapes
:22

a
add_1Addconv2d_1Variable_3/read*/
_output_shapes
:џџџџџџџџџ22
*
T0
O
relu_1Reluadd_1*
T0*/
_output_shapes
:џџџџџџџџџ22

g
Kernel_2/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=

Kernel_2/RandomStandardNormalRandomStandardNormalKernel_2/shape*

seed *
T0*
dtype0*&
_output_shapes
:



*
seed2 
t
Kernel_2/mulMulKernel_2/RandomStandardNormalKernel_2/stddev*
T0*&
_output_shapes
:




]
Kernel_2AddKernel_2/mulKernel_2/mean*
T0*&
_output_shapes
:






Variable_4
VariableV2*
shape:



*
shared_name *
dtype0*&
_output_shapes
:



*
	container 
Њ
Variable_4/AssignAssign
Variable_4Kernel_2*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
w
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*&
_output_shapes
:




г
conv2d_2Conv2Drelu_1Variable_4/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22

l
Kernel_Bias_2/shapeConst*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_2/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_2/RandomStandardNormalRandomStandardNormalKernel_Bias_2/shape*
T0*
dtype0*&
_output_shapes
:22
*
seed2 *

seed 

Kernel_Bias_2/mulMul"Kernel_Bias_2/RandomStandardNormalKernel_Bias_2/stddev*
T0*&
_output_shapes
:22

l
Kernel_Bias_2AddKernel_Bias_2/mulKernel_Bias_2/mean*
T0*&
_output_shapes
:22



Variable_5
VariableV2*
dtype0*&
_output_shapes
:22
*
	container *
shape:22
*
shared_name 
Џ
Variable_5/AssignAssign
Variable_5Kernel_Bias_2*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
:22

w
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*&
_output_shapes
:22

a
add_2Addconv2d_2Variable_5/read*
T0*/
_output_shapes
:џџџџџџџџџ22

O
relu_2Reluadd_2*
T0*/
_output_shapes
:џџџџџџџџџ22

g
Kernel_3/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=

Kernel_3/RandomStandardNormalRandomStandardNormalKernel_3/shape*
T0*
dtype0*&
_output_shapes
:



*
seed2 *

seed 
t
Kernel_3/mulMulKernel_3/RandomStandardNormalKernel_3/stddev*
T0*&
_output_shapes
:




]
Kernel_3AddKernel_3/mulKernel_3/mean*
T0*&
_output_shapes
:






Variable_6
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
Њ
Variable_6/AssignAssign
Variable_6Kernel_3*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*&
_output_shapes
:




w
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*&
_output_shapes
:



*
T0
г
conv2d_3Conv2Drelu_2Variable_6/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
l
Kernel_Bias_3/shapeConst*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_3/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_3/RandomStandardNormalRandomStandardNormalKernel_Bias_3/shape*
dtype0*&
_output_shapes
:22
*
seed2 *

seed *
T0

Kernel_Bias_3/mulMul"Kernel_Bias_3/RandomStandardNormalKernel_Bias_3/stddev*&
_output_shapes
:22
*
T0
l
Kernel_Bias_3AddKernel_Bias_3/mulKernel_Bias_3/mean*
T0*&
_output_shapes
:22



Variable_7
VariableV2*
dtype0*&
_output_shapes
:22
*
	container *
shape:22
*
shared_name 
Џ
Variable_7/AssignAssign
Variable_7Kernel_Bias_3*
validate_shape(*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_7
w
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*&
_output_shapes
:22

a
add_3Addconv2d_3Variable_7/read*/
_output_shapes
:џџџџџџџџџ22
*
T0
O
relu_3Reluadd_3*/
_output_shapes
:џџџџџџџџџ22
*
T0
g
Kernel_4/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
T
Kernel_4/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_4/RandomStandardNormalRandomStandardNormalKernel_4/shape*
T0*
dtype0*&
_output_shapes
:



*
seed2 *

seed 
t
Kernel_4/mulMulKernel_4/RandomStandardNormalKernel_4/stddev*
T0*&
_output_shapes
:




]
Kernel_4AddKernel_4/mulKernel_4/mean*
T0*&
_output_shapes
:






Variable_8
VariableV2*&
_output_shapes
:



*
	container *
shape:



*
shared_name *
dtype0
Њ
Variable_8/AssignAssign
Variable_8Kernel_4*
_class
loc:@Variable_8*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
w
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8*&
_output_shapes
:




г
conv2d_4Conv2Drelu_3Variable_8/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
l
Kernel_Bias_4/shapeConst*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_4/stddevConst*
_output_shapes
: *
valueB
 *ЭЬL=*
dtype0
І
"Kernel_Bias_4/RandomStandardNormalRandomStandardNormalKernel_Bias_4/shape*
T0*
dtype0*&
_output_shapes
:22
*
seed2 *

seed 

Kernel_Bias_4/mulMul"Kernel_Bias_4/RandomStandardNormalKernel_Bias_4/stddev*
T0*&
_output_shapes
:22

l
Kernel_Bias_4AddKernel_Bias_4/mulKernel_Bias_4/mean*
T0*&
_output_shapes
:22



Variable_9
VariableV2*&
_output_shapes
:22
*
	container *
shape:22
*
shared_name *
dtype0
Џ
Variable_9/AssignAssign
Variable_9Kernel_Bias_4*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*&
_output_shapes
:22

w
Variable_9/readIdentity
Variable_9*&
_output_shapes
:22
*
T0*
_class
loc:@Variable_9
a
add_4Addconv2d_4Variable_9/read*
T0*/
_output_shapes
:џџџџџџџџџ22

O
relu_4Reluadd_4*/
_output_shapes
:џџџџџџџџџ22
*
T0
Ѕ
Pool_4MaxPoolrelu_4*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
T0
g
Kernel_5/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_5/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=

Kernel_5/RandomStandardNormalRandomStandardNormalKernel_5/shape*&
_output_shapes
:



*
seed2 *

seed *
T0*
dtype0
t
Kernel_5/mulMulKernel_5/RandomStandardNormalKernel_5/stddev*&
_output_shapes
:



*
T0
]
Kernel_5AddKernel_5/mulKernel_5/mean*
T0*&
_output_shapes
:





Variable_10
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
­
Variable_10/AssignAssignVariable_10Kernel_5*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_10
z
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10*&
_output_shapes
:




д
conv2d_5Conv2DPool_4Variable_10/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
l
Kernel_Bias_5/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_5/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_5/RandomStandardNormalRandomStandardNormalKernel_Bias_5/shape*&
_output_shapes
:
*
seed2 *

seed *
T0*
dtype0

Kernel_Bias_5/mulMul"Kernel_Bias_5/RandomStandardNormalKernel_Bias_5/stddev*&
_output_shapes
:
*
T0
l
Kernel_Bias_5AddKernel_Bias_5/mulKernel_Bias_5/mean*&
_output_shapes
:
*
T0

Variable_11
VariableV2*&
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
В
Variable_11/AssignAssignVariable_11Kernel_Bias_5*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*&
_output_shapes
:

z
Variable_11/readIdentityVariable_11*
_class
loc:@Variable_11*&
_output_shapes
:
*
T0
b
add_5Addconv2d_5Variable_11/read*/
_output_shapes
:џџџџџџџџџ
*
T0
O
relu_5Reluadd_5*
T0*/
_output_shapes
:џџџџџџџџџ

g
Kernel_6/shapeConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   
R
Kernel_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_6/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_6/RandomStandardNormalRandomStandardNormalKernel_6/shape*
dtype0*&
_output_shapes
:



*
seed2 *

seed *
T0
t
Kernel_6/mulMulKernel_6/RandomStandardNormalKernel_6/stddev*&
_output_shapes
:



*
T0
]
Kernel_6AddKernel_6/mulKernel_6/mean*&
_output_shapes
:



*
T0

Variable_12
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
­
Variable_12/AssignAssignVariable_12Kernel_6*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*&
_output_shapes
:




z
Variable_12/readIdentityVariable_12*
T0*
_class
loc:@Variable_12*&
_output_shapes
:




д
conv2d_6Conv2Drelu_5Variable_12/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ

l
Kernel_Bias_6/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_6/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_6/RandomStandardNormalRandomStandardNormalKernel_Bias_6/shape*
T0*
dtype0*&
_output_shapes
:
*
seed2 *

seed 

Kernel_Bias_6/mulMul"Kernel_Bias_6/RandomStandardNormalKernel_Bias_6/stddev*
T0*&
_output_shapes
:

l
Kernel_Bias_6AddKernel_Bias_6/mulKernel_Bias_6/mean*
T0*&
_output_shapes
:


Variable_13
VariableV2*&
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
В
Variable_13/AssignAssignVariable_13Kernel_Bias_6*
T0*
_class
loc:@Variable_13*
validate_shape(*&
_output_shapes
:
*
use_locking(
z
Variable_13/readIdentityVariable_13*
T0*
_class
loc:@Variable_13*&
_output_shapes
:

b
add_6Addconv2d_6Variable_13/read*
T0*/
_output_shapes
:џџџџџџџџџ

O
relu_6Reluadd_6*/
_output_shapes
:џџџџџџџџџ
*
T0
g
Kernel_7/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_7/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_7/RandomStandardNormalRandomStandardNormalKernel_7/shape*&
_output_shapes
:



*
seed2 *

seed *
T0*
dtype0
t
Kernel_7/mulMulKernel_7/RandomStandardNormalKernel_7/stddev*&
_output_shapes
:



*
T0
]
Kernel_7AddKernel_7/mulKernel_7/mean*&
_output_shapes
:



*
T0

Variable_14
VariableV2*&
_output_shapes
:



*
	container *
shape:



*
shared_name *
dtype0
­
Variable_14/AssignAssignVariable_14Kernel_7*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(*&
_output_shapes
:




z
Variable_14/readIdentityVariable_14*
T0*
_class
loc:@Variable_14*&
_output_shapes
:




д
conv2d_7Conv2Drelu_6Variable_14/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
	dilations

l
Kernel_Bias_7/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_7/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_7/RandomStandardNormalRandomStandardNormalKernel_Bias_7/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 

Kernel_Bias_7/mulMul"Kernel_Bias_7/RandomStandardNormalKernel_Bias_7/stddev*
T0*&
_output_shapes
:

l
Kernel_Bias_7AddKernel_Bias_7/mulKernel_Bias_7/mean*&
_output_shapes
:
*
T0

Variable_15
VariableV2*
dtype0*&
_output_shapes
:
*
	container *
shape:
*
shared_name 
В
Variable_15/AssignAssignVariable_15Kernel_Bias_7*
T0*
_class
loc:@Variable_15*
validate_shape(*&
_output_shapes
:
*
use_locking(
z
Variable_15/readIdentityVariable_15*
T0*
_class
loc:@Variable_15*&
_output_shapes
:

b
add_7Addconv2d_7Variable_15/read*
T0*/
_output_shapes
:џџџџџџџџџ

O
relu_7Reluadd_7*
T0*/
_output_shapes
:џџџџџџџџџ

g
Kernel_8/shapeConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   
R
Kernel_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_8/stddevConst*
_output_shapes
: *
valueB
 *ЭЬL=*
dtype0

Kernel_8/RandomStandardNormalRandomStandardNormalKernel_8/shape*
T0*
dtype0*&
_output_shapes
:



*
seed2 *

seed 
t
Kernel_8/mulMulKernel_8/RandomStandardNormalKernel_8/stddev*
T0*&
_output_shapes
:




]
Kernel_8AddKernel_8/mulKernel_8/mean*
T0*&
_output_shapes
:





Variable_16
VariableV2*
shape:



*
shared_name *
dtype0*&
_output_shapes
:



*
	container 
­
Variable_16/AssignAssignVariable_16Kernel_8*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(
z
Variable_16/readIdentityVariable_16*
T0*
_class
loc:@Variable_16*&
_output_shapes
:




д
conv2d_8Conv2Drelu_7Variable_16/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ

l
Kernel_Bias_8/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_8/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=
І
"Kernel_Bias_8/RandomStandardNormalRandomStandardNormalKernel_Bias_8/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 

Kernel_Bias_8/mulMul"Kernel_Bias_8/RandomStandardNormalKernel_Bias_8/stddev*
T0*&
_output_shapes
:

l
Kernel_Bias_8AddKernel_Bias_8/mulKernel_Bias_8/mean*&
_output_shapes
:
*
T0

Variable_17
VariableV2*
shape:
*
shared_name *
dtype0*&
_output_shapes
:
*
	container 
В
Variable_17/AssignAssignVariable_17Kernel_Bias_8*
_class
loc:@Variable_17*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0
z
Variable_17/readIdentityVariable_17*
T0*
_class
loc:@Variable_17*&
_output_shapes
:

b
add_8Addconv2d_8Variable_17/read*/
_output_shapes
:џџџџџџџџџ
*
T0
O
relu_8Reluadd_8*/
_output_shapes
:џџџџџџџџџ
*
T0
Ѕ
Pool_8MaxPoolrelu_8*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ

g
Kernel_9/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_9/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
T
Kernel_9/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_9/RandomStandardNormalRandomStandardNormalKernel_9/shape*

seed *
T0*
dtype0*&
_output_shapes
:



*
seed2 
t
Kernel_9/mulMulKernel_9/RandomStandardNormalKernel_9/stddev*
T0*&
_output_shapes
:




]
Kernel_9AddKernel_9/mulKernel_9/mean*&
_output_shapes
:



*
T0

Variable_18
VariableV2*
shared_name *
dtype0*&
_output_shapes
:



*
	container *
shape:




­
Variable_18/AssignAssignVariable_18Kernel_9*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
z
Variable_18/readIdentityVariable_18*
T0*
_class
loc:@Variable_18*&
_output_shapes
:




д
conv2d_9Conv2DPool_8Variable_18/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0
l
Kernel_Bias_9/shapeConst*
_output_shapes
:*%
valueB"         
   *
dtype0
W
Kernel_Bias_9/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_9/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=
І
"Kernel_Bias_9/RandomStandardNormalRandomStandardNormalKernel_Bias_9/shape*
dtype0*&
_output_shapes
:
*
seed2 *

seed *
T0

Kernel_Bias_9/mulMul"Kernel_Bias_9/RandomStandardNormalKernel_Bias_9/stddev*
T0*&
_output_shapes
:

l
Kernel_Bias_9AddKernel_Bias_9/mulKernel_Bias_9/mean*
T0*&
_output_shapes
:


Variable_19
VariableV2*
dtype0*&
_output_shapes
:
*
	container *
shape:
*
shared_name 
В
Variable_19/AssignAssignVariable_19Kernel_Bias_9*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*&
_output_shapes
:

z
Variable_19/readIdentityVariable_19*
T0*
_class
loc:@Variable_19*&
_output_shapes
:

b
add_9Addconv2d_9Variable_19/read*/
_output_shapes
:џџџџџџџџџ
*
T0
O
relu_9Reluadd_9*
T0*/
_output_shapes
:џџџџџџџџџ

h
Kernel_10/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
S
Kernel_10/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
U
Kernel_10/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_10/RandomStandardNormalRandomStandardNormalKernel_10/shape*
T0*
dtype0*&
_output_shapes
:



*
seed2 *

seed 
w
Kernel_10/mulMulKernel_10/RandomStandardNormalKernel_10/stddev*
T0*&
_output_shapes
:




`
	Kernel_10AddKernel_10/mulKernel_10/mean*
T0*&
_output_shapes
:





Variable_20
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
Ў
Variable_20/AssignAssignVariable_20	Kernel_10*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*&
_output_shapes
:




z
Variable_20/readIdentityVariable_20*
T0*
_class
loc:@Variable_20*&
_output_shapes
:




е
	conv2d_10Conv2Drelu_9Variable_20/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0
m
Kernel_Bias_10/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         
   
X
Kernel_Bias_10/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
Kernel_Bias_10/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
Ј
#Kernel_Bias_10/RandomStandardNormalRandomStandardNormalKernel_Bias_10/shape*
T0*
dtype0*&
_output_shapes
:
*
seed2 *

seed 

Kernel_Bias_10/mulMul#Kernel_Bias_10/RandomStandardNormalKernel_Bias_10/stddev*&
_output_shapes
:
*
T0
o
Kernel_Bias_10AddKernel_Bias_10/mulKernel_Bias_10/mean*&
_output_shapes
:
*
T0

Variable_21
VariableV2*&
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
Г
Variable_21/AssignAssignVariable_21Kernel_Bias_10*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(*&
_output_shapes
:

z
Variable_21/readIdentityVariable_21*&
_output_shapes
:
*
T0*
_class
loc:@Variable_21
d
add_10Add	conv2d_10Variable_21/read*/
_output_shapes
:џџџџџџџџџ
*
T0
Q
relu_10Reluadd_10*/
_output_shapes
:џџџџџџџџџ
*
T0
h
Kernel_11/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
S
Kernel_11/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
U
Kernel_11/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_11/RandomStandardNormalRandomStandardNormalKernel_11/shape*
dtype0*&
_output_shapes
:



*
seed2 *

seed *
T0
w
Kernel_11/mulMulKernel_11/RandomStandardNormalKernel_11/stddev*&
_output_shapes
:



*
T0
`
	Kernel_11AddKernel_11/mulKernel_11/mean*
T0*&
_output_shapes
:





Variable_22
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
Ў
Variable_22/AssignAssignVariable_22	Kernel_11*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_22
z
Variable_22/readIdentityVariable_22*
_class
loc:@Variable_22*&
_output_shapes
:



*
T0
ж
	conv2d_11Conv2Drelu_10Variable_22/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ

m
Kernel_Bias_11/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
X
Kernel_Bias_11/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
Kernel_Bias_11/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
Ј
#Kernel_Bias_11/RandomStandardNormalRandomStandardNormalKernel_Bias_11/shape*
T0*
dtype0*&
_output_shapes
:
*
seed2 *

seed 

Kernel_Bias_11/mulMul#Kernel_Bias_11/RandomStandardNormalKernel_Bias_11/stddev*
T0*&
_output_shapes
:

o
Kernel_Bias_11AddKernel_Bias_11/mulKernel_Bias_11/mean*&
_output_shapes
:
*
T0

Variable_23
VariableV2*&
_output_shapes
:
*
	container *
shape:
*
shared_name *
dtype0
Г
Variable_23/AssignAssignVariable_23Kernel_Bias_11*
use_locking(*
T0*
_class
loc:@Variable_23*
validate_shape(*&
_output_shapes
:

z
Variable_23/readIdentityVariable_23*
T0*
_class
loc:@Variable_23*&
_output_shapes
:

d
add_11Add	conv2d_11Variable_23/read*
T0*/
_output_shapes
:џџџџџџџџџ

Q
relu_11Reluadd_11*
T0*/
_output_shapes
:џџџџџџџџџ

h
Kernel_12/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
S
Kernel_12/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
U
Kernel_12/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=

Kernel_12/RandomStandardNormalRandomStandardNormalKernel_12/shape*
dtype0*&
_output_shapes
:



*
seed2 *

seed *
T0
w
Kernel_12/mulMulKernel_12/RandomStandardNormalKernel_12/stddev*&
_output_shapes
:



*
T0
`
	Kernel_12AddKernel_12/mulKernel_12/mean*
T0*&
_output_shapes
:





Variable_24
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
Ў
Variable_24/AssignAssignVariable_24	Kernel_12*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_24
z
Variable_24/readIdentityVariable_24*
T0*
_class
loc:@Variable_24*&
_output_shapes
:




ж
	conv2d_12Conv2Drelu_11Variable_24/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
m
Kernel_Bias_12/shapeConst*
_output_shapes
:*%
valueB"         
   *
dtype0
X
Kernel_Bias_12/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
Kernel_Bias_12/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
Ј
#Kernel_Bias_12/RandomStandardNormalRandomStandardNormalKernel_Bias_12/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 

Kernel_Bias_12/mulMul#Kernel_Bias_12/RandomStandardNormalKernel_Bias_12/stddev*
T0*&
_output_shapes
:

o
Kernel_Bias_12AddKernel_Bias_12/mulKernel_Bias_12/mean*
T0*&
_output_shapes
:


Variable_25
VariableV2*
dtype0*&
_output_shapes
:
*
	container *
shape:
*
shared_name 
Г
Variable_25/AssignAssignVariable_25Kernel_Bias_12*
T0*
_class
loc:@Variable_25*
validate_shape(*&
_output_shapes
:
*
use_locking(
z
Variable_25/readIdentityVariable_25*&
_output_shapes
:
*
T0*
_class
loc:@Variable_25
d
add_12Add	conv2d_12Variable_25/read*
T0*/
_output_shapes
:џџџџџџџџџ

Q
relu_12Reluadd_12*
T0*/
_output_shapes
:џџџџџџџџџ

Ї
Pool_12MaxPoolrelu_12*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
T0*
strides
*
data_formatNHWC*
ksize

^
Reshape/shapeConst*
_output_shapes
:*
valueB"   џџџџ*
dtype0
j
ReshapeReshapePool_12Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
`
FC_Weight/shapeConst*
valueB"ъ  	   *
dtype0*
_output_shapes
:
S
FC_Weight/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
U
FC_Weight/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

FC_Weight/RandomStandardNormalRandomStandardNormalFC_Weight/shape*

seed *
T0*
dtype0*
_output_shapes
:	ъ	*
seed2 
p
FC_Weight/mulMulFC_Weight/RandomStandardNormalFC_Weight/stddev*
T0*
_output_shapes
:	ъ	
Y
	FC_WeightAddFC_Weight/mulFC_Weight/mean*
T0*
_output_shapes
:	ъ	

Variable_26
VariableV2*
shape:	ъ	*
shared_name *
dtype0*
_output_shapes
:	ъ	*
	container 
Ї
Variable_26/AssignAssignVariable_26	FC_Weight*
use_locking(*
T0*
_class
loc:@Variable_26*
validate_shape(*
_output_shapes
:	ъ	
s
Variable_26/readIdentityVariable_26*
_class
loc:@Variable_26*
_output_shapes
:	ъ	*
T0
^
FC_Bias/shapeConst*
valueB"   	   *
dtype0*
_output_shapes
:
Q
FC_Bias/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
S
FC_Bias/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

FC_Bias/RandomStandardNormalRandomStandardNormalFC_Bias/shape*
dtype0*
_output_shapes

:	*
seed2 *

seed *
T0
i
FC_Bias/mulMulFC_Bias/RandomStandardNormalFC_Bias/stddev*
_output_shapes

:	*
T0
R
FC_BiasAddFC_Bias/mulFC_Bias/mean*
T0*
_output_shapes

:	

Variable_27
VariableV2*
shape
:	*
shared_name *
dtype0*
_output_shapes

:	*
	container 
Є
Variable_27/AssignAssignVariable_27FC_Bias*
_output_shapes

:	*
use_locking(*
T0*
_class
loc:@Variable_27*
validate_shape(
r
Variable_27/readIdentityVariable_27*
T0*
_class
loc:@Variable_27*
_output_shapes

:	
}
	FC_MatMulMatMulReshapeVariable_26/read*
_output_shapes

:	*
transpose_a( *
transpose_b( *
T0
S
add_13Add	FC_MatMulVariable_27/read*
T0*
_output_shapes

:	
V
dropout/keep_probConst*
_output_shapes
: *
valueB
 *33s?*
dtype0
^
dropout/ShapeConst*
valueB"   	   *
dtype0*
_output_shapes
:
_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
dtype0*
_output_shapes

:	*
seed2 *

seed *
T0
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*
_output_shapes

:	
~
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
_output_shapes

:	*
T0
f
dropout/addAdddropout/keep_probdropout/random_uniform*
_output_shapes

:	*
T0
L
dropout/FloorFloordropout/add*
T0*
_output_shapes

:	
Z
dropout/divRealDivadd_13dropout/keep_prob*
T0*
_output_shapes

:	
W
dropout/mulMuldropout/divdropout/Floor*
_output_shapes

:	*
T0
Y
Label_Maker/on_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Z
Label_Maker/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
S
Label_Maker/depthConst*
value	B :	*
dtype0*
_output_shapes
: 
І
Label_MakerOneHotPlaceholder_yLabel_Maker/depthLabel_Maker/on_valueLabel_Maker/off_value*
T0*
TI0*
axisџџџџџџџџџ*
_output_shapes
:
S
Loss_SOFTMAX/RankConst*
_output_shapes
: *
value	B :*
dtype0
c
Loss_SOFTMAX/ShapeConst*
valueB"   	   *
dtype0*
_output_shapes
:
U
Loss_SOFTMAX/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
e
Loss_SOFTMAX/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   	   
T
Loss_SOFTMAX/Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
a
Loss_SOFTMAX/SubSubLoss_SOFTMAX/Rank_1Loss_SOFTMAX/Sub/y*
T0*
_output_shapes
: 
l
Loss_SOFTMAX/Slice/beginPackLoss_SOFTMAX/Sub*
T0*

axis *
N*
_output_shapes
:
a
Loss_SOFTMAX/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

Loss_SOFTMAX/SliceSliceLoss_SOFTMAX/Shape_1Loss_SOFTMAX/Slice/beginLoss_SOFTMAX/Slice/size*
Index0*
T0*
_output_shapes
:
o
Loss_SOFTMAX/concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Z
Loss_SOFTMAX/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
Loss_SOFTMAX/concatConcatV2Loss_SOFTMAX/concat/values_0Loss_SOFTMAX/SliceLoss_SOFTMAX/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
s
Loss_SOFTMAX/ReshapeReshapeadd_13Loss_SOFTMAX/concat*
T0*
Tshape0*
_output_shapes

:	
I
Loss_SOFTMAX/Rank_2RankLabel_Maker*
_output_shapes
: *
T0
h
Loss_SOFTMAX/Shape_2ShapeLabel_Maker*#
_output_shapes
:џџџџџџџџџ*
T0*
out_type0
V
Loss_SOFTMAX/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
e
Loss_SOFTMAX/Sub_1SubLoss_SOFTMAX/Rank_2Loss_SOFTMAX/Sub_1/y*
_output_shapes
: *
T0
p
Loss_SOFTMAX/Slice_1/beginPackLoss_SOFTMAX/Sub_1*
T0*

axis *
N*
_output_shapes
:
c
Loss_SOFTMAX/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

Loss_SOFTMAX/Slice_1SliceLoss_SOFTMAX/Shape_2Loss_SOFTMAX/Slice_1/beginLoss_SOFTMAX/Slice_1/size*
_output_shapes
:*
Index0*
T0
q
Loss_SOFTMAX/concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
\
Loss_SOFTMAX/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
­
Loss_SOFTMAX/concat_1ConcatV2Loss_SOFTMAX/concat_1/values_0Loss_SOFTMAX/Slice_1Loss_SOFTMAX/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

Loss_SOFTMAX/Reshape_1ReshapeLabel_MakerLoss_SOFTMAX/concat_1*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

Loss_SOFTMAXSoftmaxCrossEntropyWithLogitsLoss_SOFTMAX/ReshapeLoss_SOFTMAX/Reshape_1*$
_output_shapes
::	*
T0
V
Loss_SOFTMAX/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
c
Loss_SOFTMAX/Sub_2SubLoss_SOFTMAX/RankLoss_SOFTMAX/Sub_2/y*
T0*
_output_shapes
: 
d
Loss_SOFTMAX/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
o
Loss_SOFTMAX/Slice_2/sizePackLoss_SOFTMAX/Sub_2*
T0*

axis *
N*
_output_shapes
:
Ѓ
Loss_SOFTMAX/Slice_2SliceLoss_SOFTMAX/ShapeLoss_SOFTMAX/Slice_2/beginLoss_SOFTMAX/Slice_2/size*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ
x
Loss_SOFTMAX/Reshape_2ReshapeLoss_SOFTMAXLoss_SOFTMAX/Slice_2*
_output_shapes
:*
T0*
Tshape0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
p
Reduce_MeanMeanLoss_SOFTMAX/Reshape_2Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
r
(gradients/Reduce_Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

"gradients/Reduce_Mean_grad/ReshapeReshapegradients/Fill(gradients/Reduce_Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
j
 gradients/Reduce_Mean_grad/ConstConst*
valueB:*
dtype0*
_output_shapes
:
Є
gradients/Reduce_Mean_grad/TileTile"gradients/Reduce_Mean_grad/Reshape gradients/Reduce_Mean_grad/Const*
T0*
_output_shapes
:*

Tmultiples0
g
"gradients/Reduce_Mean_grad/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"gradients/Reduce_Mean_grad/truedivRealDivgradients/Reduce_Mean_grad/Tile"gradients/Reduce_Mean_grad/Const_1*
T0*
_output_shapes
:
u
+gradients/Loss_SOFTMAX/Reshape_2_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
М
-gradients/Loss_SOFTMAX/Reshape_2_grad/ReshapeReshape"gradients/Reduce_Mean_grad/truediv+gradients/Loss_SOFTMAX/Reshape_2_grad/Shape*
Tshape0*
_output_shapes
:*
T0
Z
gradients/zeros_like	ZerosLikeLoss_SOFTMAX:1*
T0*
_output_shapes

:	
u
*gradients/Loss_SOFTMAX_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ф
&gradients/Loss_SOFTMAX_grad/ExpandDims
ExpandDims-gradients/Loss_SOFTMAX/Reshape_2_grad/Reshape*gradients/Loss_SOFTMAX_grad/ExpandDims/dim*
_output_shapes

:*

Tdim0*
T0

gradients/Loss_SOFTMAX_grad/mulMul&gradients/Loss_SOFTMAX_grad/ExpandDimsLoss_SOFTMAX:1*
T0*
_output_shapes

:	
s
&gradients/Loss_SOFTMAX_grad/LogSoftmax
LogSoftmaxLoss_SOFTMAX/Reshape*
_output_shapes

:	*
T0
w
gradients/Loss_SOFTMAX_grad/NegNeg&gradients/Loss_SOFTMAX_grad/LogSoftmax*
T0*
_output_shapes

:	
w
,gradients/Loss_SOFTMAX_grad/ExpandDims_1/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ш
(gradients/Loss_SOFTMAX_grad/ExpandDims_1
ExpandDims-gradients/Loss_SOFTMAX/Reshape_2_grad/Reshape,gradients/Loss_SOFTMAX_grad/ExpandDims_1/dim*
_output_shapes

:*

Tdim0*
T0

!gradients/Loss_SOFTMAX_grad/mul_1Mul(gradients/Loss_SOFTMAX_grad/ExpandDims_1gradients/Loss_SOFTMAX_grad/Neg*
_output_shapes

:	*
T0
z
,gradients/Loss_SOFTMAX_grad/tuple/group_depsNoOp ^gradients/Loss_SOFTMAX_grad/mul"^gradients/Loss_SOFTMAX_grad/mul_1
э
4gradients/Loss_SOFTMAX_grad/tuple/control_dependencyIdentitygradients/Loss_SOFTMAX_grad/mul-^gradients/Loss_SOFTMAX_grad/tuple/group_deps*
_output_shapes

:	*
T0*2
_class(
&$loc:@gradients/Loss_SOFTMAX_grad/mul
ѓ
6gradients/Loss_SOFTMAX_grad/tuple/control_dependency_1Identity!gradients/Loss_SOFTMAX_grad/mul_1-^gradients/Loss_SOFTMAX_grad/tuple/group_deps*
_output_shapes

:	*
T0*4
_class*
(&loc:@gradients/Loss_SOFTMAX_grad/mul_1
z
)gradients/Loss_SOFTMAX/Reshape_grad/ShapeConst*
valueB"   	   *
dtype0*
_output_shapes
:
Ю
+gradients/Loss_SOFTMAX/Reshape_grad/ReshapeReshape4gradients/Loss_SOFTMAX_grad/tuple/control_dependency)gradients/Loss_SOFTMAX/Reshape_grad/Shape*
Tshape0*
_output_shapes

:	*
T0
\
&gradients/add_13_grad/tuple/group_depsNoOp,^gradients/Loss_SOFTMAX/Reshape_grad/Reshape
љ
.gradients/add_13_grad/tuple/control_dependencyIdentity+gradients/Loss_SOFTMAX/Reshape_grad/Reshape'^gradients/add_13_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Loss_SOFTMAX/Reshape_grad/Reshape*
_output_shapes

:	
ћ
0gradients/add_13_grad/tuple/control_dependency_1Identity+gradients/Loss_SOFTMAX/Reshape_grad/Reshape'^gradients/add_13_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Loss_SOFTMAX/Reshape_grad/Reshape*
_output_shapes

:	
Л
gradients/FC_MatMul_grad/MatMulMatMul.gradients/add_13_grad/tuple/control_dependencyVariable_26/read*
T0*
_output_shapes
:	ъ*
transpose_a( *
transpose_b(
М
!gradients/FC_MatMul_grad/MatMul_1MatMulReshape.gradients/add_13_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ	*
transpose_a(*
transpose_b( 
w
)gradients/FC_MatMul_grad/tuple/group_depsNoOp ^gradients/FC_MatMul_grad/MatMul"^gradients/FC_MatMul_grad/MatMul_1
ш
1gradients/FC_MatMul_grad/tuple/control_dependencyIdentitygradients/FC_MatMul_grad/MatMul*^gradients/FC_MatMul_grad/tuple/group_deps*2
_class(
&$loc:@gradients/FC_MatMul_grad/MatMul*
_output_shapes
:	ъ*
T0
ю
3gradients/FC_MatMul_grad/tuple/control_dependency_1Identity!gradients/FC_MatMul_grad/MatMul_1*^gradients/FC_MatMul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/FC_MatMul_grad/MatMul_1*
_output_shapes
:	ъ	
c
gradients/Reshape_grad/ShapeShapePool_12*
_output_shapes
:*
T0*
out_type0
Й
gradients/Reshape_grad/ReshapeReshape1gradients/FC_MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0*&
_output_shapes
:

я
"gradients/Pool_12_grad/MaxPoolGradMaxPoolGradrelu_12Pool_12gradients/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
T0*
strides
*
data_formatNHWC

gradients/relu_12_grad/ReluGradReluGrad"gradients/Pool_12_grad/MaxPoolGradrelu_12*
T0*/
_output_shapes
:џџџџџџџџџ

d
gradients/add_12_grad/ShapeShape	conv2d_12*
_output_shapes
:*
T0*
out_type0
v
gradients/add_12_grad/Shape_1Const*
_output_shapes
:*%
valueB"         
   *
dtype0
Н
+gradients/add_12_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_12_grad/Shapegradients/add_12_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ў
gradients/add_12_grad/SumSumgradients/relu_12_grad/ReluGrad+gradients/add_12_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ј
gradients/add_12_grad/ReshapeReshapegradients/add_12_grad/Sumgradients/add_12_grad/Shape*/
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0
В
gradients/add_12_grad/Sum_1Sumgradients/relu_12_grad/ReluGrad-gradients/add_12_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_12_grad/Reshape_1Reshapegradients/add_12_grad/Sum_1gradients/add_12_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

p
&gradients/add_12_grad/tuple/group_depsNoOp^gradients/add_12_grad/Reshape ^gradients/add_12_grad/Reshape_1
ю
.gradients/add_12_grad/tuple/control_dependencyIdentitygradients/add_12_grad/Reshape'^gradients/add_12_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_12_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ы
0gradients/add_12_grad/tuple/control_dependency_1Identitygradients/add_12_grad/Reshape_1'^gradients/add_12_grad/tuple/group_deps*2
_class(
&$loc:@gradients/add_12_grad/Reshape_1*&
_output_shapes
:
*
T0

gradients/conv2d_12_grad/ShapeNShapeNrelu_11Variable_24/read*
T0*
out_type0*
N* 
_output_shapes
::
w
gradients/conv2d_12_grad/ConstConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
щ
,gradients/conv2d_12_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_12_grad/ShapeNVariable_24/read.gradients/add_12_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Н
-gradients/conv2d_12_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_11gradients/conv2d_12_grad/Const.gradients/add_12_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

)gradients/conv2d_12_grad/tuple/group_depsNoOp-^gradients/conv2d_12_grad/Conv2DBackpropInput.^gradients/conv2d_12_grad/Conv2DBackpropFilter

1gradients/conv2d_12_grad/tuple/control_dependencyIdentity,gradients/conv2d_12_grad/Conv2DBackpropInput*^gradients/conv2d_12_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_12_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


3gradients/conv2d_12_grad/tuple/control_dependency_1Identity-gradients/conv2d_12_grad/Conv2DBackpropFilter*^gradients/conv2d_12_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/conv2d_12_grad/Conv2DBackpropFilter*&
_output_shapes
:




Ё
gradients/relu_11_grad/ReluGradReluGrad1gradients/conv2d_12_grad/tuple/control_dependencyrelu_11*
T0*/
_output_shapes
:џџџџџџџџџ

d
gradients/add_11_grad/ShapeShape	conv2d_11*
T0*
out_type0*
_output_shapes
:
v
gradients/add_11_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
Н
+gradients/add_11_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_11_grad/Shapegradients/add_11_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
gradients/add_11_grad/SumSumgradients/relu_11_grad/ReluGrad+gradients/add_11_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ј
gradients/add_11_grad/ReshapeReshapegradients/add_11_grad/Sumgradients/add_11_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ

В
gradients/add_11_grad/Sum_1Sumgradients/relu_11_grad/ReluGrad-gradients/add_11_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_11_grad/Reshape_1Reshapegradients/add_11_grad/Sum_1gradients/add_11_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

p
&gradients/add_11_grad/tuple/group_depsNoOp^gradients/add_11_grad/Reshape ^gradients/add_11_grad/Reshape_1
ю
.gradients/add_11_grad/tuple/control_dependencyIdentitygradients/add_11_grad/Reshape'^gradients/add_11_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*0
_class&
$"loc:@gradients/add_11_grad/Reshape
ы
0gradients/add_11_grad/tuple/control_dependency_1Identitygradients/add_11_grad/Reshape_1'^gradients/add_11_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_11_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_11_grad/ShapeNShapeNrelu_10Variable_22/read*
T0*
out_type0*
N* 
_output_shapes
::
w
gradients/conv2d_11_grad/ConstConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
щ
,gradients/conv2d_11_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_11_grad/ShapeNVariable_22/read.gradients/add_11_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Н
-gradients/conv2d_11_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_10gradients/conv2d_11_grad/Const.gradients/add_11_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0

)gradients/conv2d_11_grad/tuple/group_depsNoOp-^gradients/conv2d_11_grad/Conv2DBackpropInput.^gradients/conv2d_11_grad/Conv2DBackpropFilter

1gradients/conv2d_11_grad/tuple/control_dependencyIdentity,gradients/conv2d_11_grad/Conv2DBackpropInput*^gradients/conv2d_11_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_11_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


3gradients/conv2d_11_grad/tuple/control_dependency_1Identity-gradients/conv2d_11_grad/Conv2DBackpropFilter*^gradients/conv2d_11_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/conv2d_11_grad/Conv2DBackpropFilter*&
_output_shapes
:




Ё
gradients/relu_10_grad/ReluGradReluGrad1gradients/conv2d_11_grad/tuple/control_dependencyrelu_10*/
_output_shapes
:џџџџџџџџџ
*
T0
d
gradients/add_10_grad/ShapeShape	conv2d_10*
_output_shapes
:*
T0*
out_type0
v
gradients/add_10_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         
   
Н
+gradients/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_10_grad/Shapegradients/add_10_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
gradients/add_10_grad/SumSumgradients/relu_10_grad/ReluGrad+gradients/add_10_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ј
gradients/add_10_grad/ReshapeReshapegradients/add_10_grad/Sumgradients/add_10_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ
*
T0
В
gradients/add_10_grad/Sum_1Sumgradients/relu_10_grad/ReluGrad-gradients/add_10_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ѕ
gradients/add_10_grad/Reshape_1Reshapegradients/add_10_grad/Sum_1gradients/add_10_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

p
&gradients/add_10_grad/tuple/group_depsNoOp^gradients/add_10_grad/Reshape ^gradients/add_10_grad/Reshape_1
ю
.gradients/add_10_grad/tuple/control_dependencyIdentitygradients/add_10_grad/Reshape'^gradients/add_10_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_10_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ы
0gradients/add_10_grad/tuple/control_dependency_1Identitygradients/add_10_grad/Reshape_1'^gradients/add_10_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_10_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_10_grad/ShapeNShapeNrelu_9Variable_20/read*
N* 
_output_shapes
::*
T0*
out_type0
w
gradients/conv2d_10_grad/ConstConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
щ
,gradients/conv2d_10_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_10_grad/ShapeNVariable_20/read.gradients/add_10_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
М
-gradients/conv2d_10_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_9gradients/conv2d_10_grad/Const.gradients/add_10_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0

)gradients/conv2d_10_grad/tuple/group_depsNoOp-^gradients/conv2d_10_grad/Conv2DBackpropInput.^gradients/conv2d_10_grad/Conv2DBackpropFilter

1gradients/conv2d_10_grad/tuple/control_dependencyIdentity,gradients/conv2d_10_grad/Conv2DBackpropInput*^gradients/conv2d_10_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*?
_class5
31loc:@gradients/conv2d_10_grad/Conv2DBackpropInput

3gradients/conv2d_10_grad/tuple/control_dependency_1Identity-gradients/conv2d_10_grad/Conv2DBackpropFilter*^gradients/conv2d_10_grad/tuple/group_deps*@
_class6
42loc:@gradients/conv2d_10_grad/Conv2DBackpropFilter*&
_output_shapes
:



*
T0

gradients/relu_9_grad/ReluGradReluGrad1gradients/conv2d_10_grad/tuple/control_dependencyrelu_9*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_9_grad/ShapeShapeconv2d_9*
T0*
out_type0*
_output_shapes
:
u
gradients/add_9_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         
   
К
*gradients/add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_9_grad/Shapegradients/add_9_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_9_grad/SumSumgradients/relu_9_grad/ReluGrad*gradients/add_9_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ѕ
gradients/add_9_grad/ReshapeReshapegradients/add_9_grad/Sumgradients/add_9_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ

Џ
gradients/add_9_grad/Sum_1Sumgradients/relu_9_grad/ReluGrad,gradients/add_9_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ђ
gradients/add_9_grad/Reshape_1Reshapegradients/add_9_grad/Sum_1gradients/add_9_grad/Shape_1*&
_output_shapes
:
*
T0*
Tshape0
m
%gradients/add_9_grad/tuple/group_depsNoOp^gradients/add_9_grad/Reshape^gradients/add_9_grad/Reshape_1
ъ
-gradients/add_9_grad/tuple/control_dependencyIdentitygradients/add_9_grad/Reshape&^gradients/add_9_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*/
_class%
#!loc:@gradients/add_9_grad/Reshape
ч
/gradients/add_9_grad/tuple/control_dependency_1Identitygradients/add_9_grad/Reshape_1&^gradients/add_9_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_9_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_9_grad/ShapeNShapeNPool_8Variable_18/read* 
_output_shapes
::*
T0*
out_type0*
N
v
gradients/conv2d_9_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
ц
+gradients/conv2d_9_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_9_grad/ShapeNVariable_18/read-gradients/add_9_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Й
,gradients/conv2d_9_grad/Conv2DBackpropFilterConv2DBackpropFilterPool_8gradients/conv2d_9_grad/Const-gradients/add_9_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0

(gradients/conv2d_9_grad/tuple/group_depsNoOp,^gradients/conv2d_9_grad/Conv2DBackpropInput-^gradients/conv2d_9_grad/Conv2DBackpropFilter

0gradients/conv2d_9_grad/tuple/control_dependencyIdentity+gradients/conv2d_9_grad/Conv2DBackpropInput)^gradients/conv2d_9_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_9_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


2gradients/conv2d_9_grad/tuple/control_dependency_1Identity,gradients/conv2d_9_grad/Conv2DBackpropFilter)^gradients/conv2d_9_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_9_grad/Conv2DBackpropFilter*&
_output_shapes
:




ў
!gradients/Pool_8_grad/MaxPoolGradMaxPoolGradrelu_8Pool_80gradients/conv2d_9_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ


gradients/relu_8_grad/ReluGradReluGrad!gradients/Pool_8_grad/MaxPoolGradrelu_8*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_8_grad/ShapeShapeconv2d_8*
out_type0*
_output_shapes
:*
T0
u
gradients/add_8_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
К
*gradients/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_8_grad/Shapegradients/add_8_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_8_grad/SumSumgradients/relu_8_grad/ReluGrad*gradients/add_8_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_8_grad/ReshapeReshapegradients/add_8_grad/Sumgradients/add_8_grad/Shape*/
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0
Џ
gradients/add_8_grad/Sum_1Sumgradients/relu_8_grad/ReluGrad,gradients/add_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ђ
gradients/add_8_grad/Reshape_1Reshapegradients/add_8_grad/Sum_1gradients/add_8_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

m
%gradients/add_8_grad/tuple/group_depsNoOp^gradients/add_8_grad/Reshape^gradients/add_8_grad/Reshape_1
ъ
-gradients/add_8_grad/tuple/control_dependencyIdentitygradients/add_8_grad/Reshape&^gradients/add_8_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_8_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ч
/gradients/add_8_grad/tuple/control_dependency_1Identitygradients/add_8_grad/Reshape_1&^gradients/add_8_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_8_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_8_grad/ShapeNShapeNrelu_7Variable_16/read*
N* 
_output_shapes
::*
T0*
out_type0
v
gradients/conv2d_8_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
ц
+gradients/conv2d_8_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_8_grad/ShapeNVariable_16/read-gradients/add_8_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Й
,gradients/conv2d_8_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_7gradients/conv2d_8_grad/Const-gradients/add_8_grad/tuple/control_dependency*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/conv2d_8_grad/tuple/group_depsNoOp,^gradients/conv2d_8_grad/Conv2DBackpropInput-^gradients/conv2d_8_grad/Conv2DBackpropFilter

0gradients/conv2d_8_grad/tuple/control_dependencyIdentity+gradients/conv2d_8_grad/Conv2DBackpropInput)^gradients/conv2d_8_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_8_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


2gradients/conv2d_8_grad/tuple/control_dependency_1Identity,gradients/conv2d_8_grad/Conv2DBackpropFilter)^gradients/conv2d_8_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_8_grad/Conv2DBackpropFilter*&
_output_shapes
:





gradients/relu_7_grad/ReluGradReluGrad0gradients/conv2d_8_grad/tuple/control_dependencyrelu_7*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_7_grad/ShapeShapeconv2d_7*
out_type0*
_output_shapes
:*
T0
u
gradients/add_7_grad/Shape_1Const*
_output_shapes
:*%
valueB"         
   *
dtype0
К
*gradients/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_7_grad/Shapegradients/add_7_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_7_grad/SumSumgradients/relu_7_grad/ReluGrad*gradients/add_7_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_7_grad/ReshapeReshapegradients/add_7_grad/Sumgradients/add_7_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ
*
T0
Џ
gradients/add_7_grad/Sum_1Sumgradients/relu_7_grad/ReluGrad,gradients/add_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ђ
gradients/add_7_grad/Reshape_1Reshapegradients/add_7_grad/Sum_1gradients/add_7_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

m
%gradients/add_7_grad/tuple/group_depsNoOp^gradients/add_7_grad/Reshape^gradients/add_7_grad/Reshape_1
ъ
-gradients/add_7_grad/tuple/control_dependencyIdentitygradients/add_7_grad/Reshape&^gradients/add_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_7_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ч
/gradients/add_7_grad/tuple/control_dependency_1Identitygradients/add_7_grad/Reshape_1&^gradients/add_7_grad/tuple/group_deps*&
_output_shapes
:
*
T0*1
_class'
%#loc:@gradients/add_7_grad/Reshape_1

gradients/conv2d_7_grad/ShapeNShapeNrelu_6Variable_14/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_7_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
ц
+gradients/conv2d_7_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_7_grad/ShapeNVariable_14/read-gradients/add_7_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Й
,gradients/conv2d_7_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_6gradients/conv2d_7_grad/Const-gradients/add_7_grad/tuple/control_dependency*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/conv2d_7_grad/tuple/group_depsNoOp,^gradients/conv2d_7_grad/Conv2DBackpropInput-^gradients/conv2d_7_grad/Conv2DBackpropFilter

0gradients/conv2d_7_grad/tuple/control_dependencyIdentity+gradients/conv2d_7_grad/Conv2DBackpropInput)^gradients/conv2d_7_grad/tuple/group_deps*>
_class4
20loc:@gradients/conv2d_7_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ
*
T0

2gradients/conv2d_7_grad/tuple/control_dependency_1Identity,gradients/conv2d_7_grad/Conv2DBackpropFilter)^gradients/conv2d_7_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_7_grad/Conv2DBackpropFilter*&
_output_shapes
:





gradients/relu_6_grad/ReluGradReluGrad0gradients/conv2d_7_grad/tuple/control_dependencyrelu_6*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_6_grad/ShapeShapeconv2d_6*
T0*
out_type0*
_output_shapes
:
u
gradients/add_6_grad/Shape_1Const*
_output_shapes
:*%
valueB"         
   *
dtype0
К
*gradients/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_6_grad/Shapegradients/add_6_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_6_grad/SumSumgradients/relu_6_grad/ReluGrad*gradients/add_6_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_6_grad/ReshapeReshapegradients/add_6_grad/Sumgradients/add_6_grad/Shape*/
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0
Џ
gradients/add_6_grad/Sum_1Sumgradients/relu_6_grad/ReluGrad,gradients/add_6_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ђ
gradients/add_6_grad/Reshape_1Reshapegradients/add_6_grad/Sum_1gradients/add_6_grad/Shape_1*
Tshape0*&
_output_shapes
:
*
T0
m
%gradients/add_6_grad/tuple/group_depsNoOp^gradients/add_6_grad/Reshape^gradients/add_6_grad/Reshape_1
ъ
-gradients/add_6_grad/tuple/control_dependencyIdentitygradients/add_6_grad/Reshape&^gradients/add_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_6_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ч
/gradients/add_6_grad/tuple/control_dependency_1Identitygradients/add_6_grad/Reshape_1&^gradients/add_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_6_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_6_grad/ShapeNShapeNrelu_5Variable_12/read* 
_output_shapes
::*
T0*
out_type0*
N
v
gradients/conv2d_6_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
ц
+gradients/conv2d_6_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_6_grad/ShapeNVariable_12/read-gradients/add_6_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Й
,gradients/conv2d_6_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_5gradients/conv2d_6_grad/Const-gradients/add_6_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:





(gradients/conv2d_6_grad/tuple/group_depsNoOp,^gradients/conv2d_6_grad/Conv2DBackpropInput-^gradients/conv2d_6_grad/Conv2DBackpropFilter

0gradients/conv2d_6_grad/tuple/control_dependencyIdentity+gradients/conv2d_6_grad/Conv2DBackpropInput)^gradients/conv2d_6_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*>
_class4
20loc:@gradients/conv2d_6_grad/Conv2DBackpropInput

2gradients/conv2d_6_grad/tuple/control_dependency_1Identity,gradients/conv2d_6_grad/Conv2DBackpropFilter)^gradients/conv2d_6_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_6_grad/Conv2DBackpropFilter*&
_output_shapes
:





gradients/relu_5_grad/ReluGradReluGrad0gradients/conv2d_6_grad/tuple/control_dependencyrelu_5*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_5_grad/ShapeShapeconv2d_5*
T0*
out_type0*
_output_shapes
:
u
gradients/add_5_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
К
*gradients/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_5_grad/Shapegradients/add_5_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_5_grad/SumSumgradients/relu_5_grad/ReluGrad*gradients/add_5_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_5_grad/ReshapeReshapegradients/add_5_grad/Sumgradients/add_5_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ

Џ
gradients/add_5_grad/Sum_1Sumgradients/relu_5_grad/ReluGrad,gradients/add_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ђ
gradients/add_5_grad/Reshape_1Reshapegradients/add_5_grad/Sum_1gradients/add_5_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

m
%gradients/add_5_grad/tuple/group_depsNoOp^gradients/add_5_grad/Reshape^gradients/add_5_grad/Reshape_1
ъ
-gradients/add_5_grad/tuple/control_dependencyIdentitygradients/add_5_grad/Reshape&^gradients/add_5_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_5_grad/Reshape*/
_output_shapes
:џџџџџџџџџ
*
T0
ч
/gradients/add_5_grad/tuple/control_dependency_1Identitygradients/add_5_grad/Reshape_1&^gradients/add_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_5_grad/Reshape_1*&
_output_shapes
:
*
T0

gradients/conv2d_5_grad/ShapeNShapeNPool_4Variable_10/read*
N* 
_output_shapes
::*
T0*
out_type0
v
gradients/conv2d_5_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
ц
+gradients/conv2d_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_5_grad/ShapeNVariable_10/read-gradients/add_5_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Й
,gradients/conv2d_5_grad/Conv2DBackpropFilterConv2DBackpropFilterPool_4gradients/conv2d_5_grad/Const-gradients/add_5_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:





(gradients/conv2d_5_grad/tuple/group_depsNoOp,^gradients/conv2d_5_grad/Conv2DBackpropInput-^gradients/conv2d_5_grad/Conv2DBackpropFilter

0gradients/conv2d_5_grad/tuple/control_dependencyIdentity+gradients/conv2d_5_grad/Conv2DBackpropInput)^gradients/conv2d_5_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*>
_class4
20loc:@gradients/conv2d_5_grad/Conv2DBackpropInput

2gradients/conv2d_5_grad/tuple/control_dependency_1Identity,gradients/conv2d_5_grad/Conv2DBackpropFilter)^gradients/conv2d_5_grad/tuple/group_deps*?
_class5
31loc:@gradients/conv2d_5_grad/Conv2DBackpropFilter*&
_output_shapes
:



*
T0
ў
!gradients/Pool_4_grad/MaxPoolGradMaxPoolGradrelu_4Pool_40gradients/conv2d_5_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22


gradients/relu_4_grad/ReluGradReluGrad!gradients/Pool_4_grad/MaxPoolGradrelu_4*
T0*/
_output_shapes
:џџџџџџџџџ22

b
gradients/add_4_grad/ShapeShapeconv2d_4*
out_type0*
_output_shapes
:*
T0
u
gradients/add_4_grad/Shape_1Const*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
К
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_4_grad/SumSumgradients/relu_4_grad/ReluGrad*gradients/add_4_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*/
_output_shapes
:џџџџџџџџџ22
*
T0*
Tshape0
Џ
gradients/add_4_grad/Sum_1Sumgradients/relu_4_grad/ReluGrad,gradients/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ђ
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:22

m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
ъ
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_4_grad/Reshape*/
_output_shapes
:џџџџџџџџџ22
*
T0
ч
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*&
_output_shapes
:22


gradients/conv2d_4_grad/ShapeNShapeNrelu_3Variable_8/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_4_grad/ConstConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
х
+gradients/conv2d_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_4_grad/ShapeNVariable_8/read-gradients/add_4_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
Й
,gradients/conv2d_4_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_3gradients/conv2d_4_grad/Const-gradients/add_4_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:





(gradients/conv2d_4_grad/tuple/group_depsNoOp,^gradients/conv2d_4_grad/Conv2DBackpropInput-^gradients/conv2d_4_grad/Conv2DBackpropFilter

0gradients/conv2d_4_grad/tuple/control_dependencyIdentity+gradients/conv2d_4_grad/Conv2DBackpropInput)^gradients/conv2d_4_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ22
*
T0*>
_class4
20loc:@gradients/conv2d_4_grad/Conv2DBackpropInput

2gradients/conv2d_4_grad/tuple/control_dependency_1Identity,gradients/conv2d_4_grad/Conv2DBackpropFilter)^gradients/conv2d_4_grad/tuple/group_deps*&
_output_shapes
:



*
T0*?
_class5
31loc:@gradients/conv2d_4_grad/Conv2DBackpropFilter

gradients/relu_3_grad/ReluGradReluGrad0gradients/conv2d_4_grad/tuple/control_dependencyrelu_3*/
_output_shapes
:џџџџџџџџџ22
*
T0
b
gradients/add_3_grad/ShapeShapeconv2d_3*
_output_shapes
:*
T0*
out_type0
u
gradients/add_3_grad/Shape_1Const*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_3_grad/SumSumgradients/relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*/
_output_shapes
:џџџџџџџџџ22
*
T0*
Tshape0
Џ
gradients/add_3_grad/Sum_1Sumgradients/relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ђ
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*&
_output_shapes
:22
*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
ъ
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*/
_output_shapes
:џџџџџџџџџ22

ч
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*&
_output_shapes
:22
*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1

gradients/conv2d_3_grad/ShapeNShapeNrelu_2Variable_6/read* 
_output_shapes
::*
T0*
out_type0*
N
v
gradients/conv2d_3_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
х
+gradients/conv2d_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_3_grad/ShapeNVariable_6/read-gradients/add_3_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations

Й
,gradients/conv2d_3_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_2gradients/conv2d_3_grad/Const-gradients/add_3_grad/tuple/control_dependency*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/conv2d_3_grad/tuple/group_depsNoOp,^gradients/conv2d_3_grad/Conv2DBackpropInput-^gradients/conv2d_3_grad/Conv2DBackpropFilter

0gradients/conv2d_3_grad/tuple/control_dependencyIdentity+gradients/conv2d_3_grad/Conv2DBackpropInput)^gradients/conv2d_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_3_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ22


2gradients/conv2d_3_grad/tuple/control_dependency_1Identity,gradients/conv2d_3_grad/Conv2DBackpropFilter)^gradients/conv2d_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_3_grad/Conv2DBackpropFilter*&
_output_shapes
:





gradients/relu_2_grad/ReluGradReluGrad0gradients/conv2d_3_grad/tuple/control_dependencyrelu_2*/
_output_shapes
:џџџџџџџџџ22
*
T0
b
gradients/add_2_grad/ShapeShapeconv2d_2*
T0*
out_type0*
_output_shapes
:
u
gradients/add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"   2   2   
   
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_2_grad/SumSumgradients/relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ѕ
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ22

Џ
gradients/add_2_grad/Sum_1Sumgradients/relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ђ
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:22

m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
ъ
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*/
_output_shapes
:џџџџџџџџџ22

ч
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*&
_output_shapes
:22


gradients/conv2d_2_grad/ShapeNShapeNrelu_1Variable_4/read*
out_type0*
N* 
_output_shapes
::*
T0
v
gradients/conv2d_2_grad/ConstConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
х
+gradients/conv2d_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_2_grad/ShapeNVariable_4/read-gradients/add_2_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Й
,gradients/conv2d_2_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_1gradients/conv2d_2_grad/Const-gradients/add_2_grad/tuple/control_dependency*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/conv2d_2_grad/tuple/group_depsNoOp,^gradients/conv2d_2_grad/Conv2DBackpropInput-^gradients/conv2d_2_grad/Conv2DBackpropFilter

0gradients/conv2d_2_grad/tuple/control_dependencyIdentity+gradients/conv2d_2_grad/Conv2DBackpropInput)^gradients/conv2d_2_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ22
*
T0*>
_class4
20loc:@gradients/conv2d_2_grad/Conv2DBackpropInput

2gradients/conv2d_2_grad/tuple/control_dependency_1Identity,gradients/conv2d_2_grad/Conv2DBackpropFilter)^gradients/conv2d_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_2_grad/Conv2DBackpropFilter*&
_output_shapes
:





gradients/relu_1_grad/ReluGradReluGrad0gradients/conv2d_2_grad/tuple/control_dependencyrelu_1*
T0*/
_output_shapes
:џџџџџџџџџ22

b
gradients/add_1_grad/ShapeShapeconv2d_1*
T0*
out_type0*
_output_shapes
:
u
gradients/add_1_grad/Shape_1Const*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_1_grad/SumSumgradients/relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ѕ
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ22
*
T0
Џ
gradients/add_1_grad/Sum_1Sumgradients/relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ђ
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*&
_output_shapes
:22
*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ъ
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_1_grad/Reshape*/
_output_shapes
:џџџџџџџџџ22
*
T0
ч
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*&
_output_shapes
:22
*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1

gradients/conv2d_1_grad/ShapeNShapeNPool_0Variable_2/read* 
_output_shapes
::*
T0*
out_type0*
N
v
gradients/conv2d_1_grad/ConstConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
х
+gradients/conv2d_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Й
,gradients/conv2d_1_grad/Conv2DBackpropFilterConv2DBackpropFilterPool_0gradients/conv2d_1_grad/Const-gradients/add_1_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC

(gradients/conv2d_1_grad/tuple/group_depsNoOp,^gradients/conv2d_1_grad/Conv2DBackpropInput-^gradients/conv2d_1_grad/Conv2DBackpropFilter

0gradients/conv2d_1_grad/tuple/control_dependencyIdentity+gradients/conv2d_1_grad/Conv2DBackpropInput)^gradients/conv2d_1_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ22
*
T0*>
_class4
20loc:@gradients/conv2d_1_grad/Conv2DBackpropInput

2gradients/conv2d_1_grad/tuple/control_dependency_1Identity,gradients/conv2d_1_grad/Conv2DBackpropFilter)^gradients/conv2d_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_1_grad/Conv2DBackpropFilter*&
_output_shapes
:




ў
!gradients/Pool_0_grad/MaxPoolGradMaxPoolGradrelu_0Pool_00gradients/conv2d_1_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџdd
*
T0*
strides
*
data_formatNHWC

gradients/relu_0_grad/ReluGradReluGrad!gradients/Pool_0_grad/MaxPoolGradrelu_0*/
_output_shapes
:џџџџџџџџџdd
*
T0
`
gradients/add_grad/ShapeShapeconv2d_0*
_output_shapes
:*
T0*
out_type0
s
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"   d   d   
   
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ї
gradients/add_grad/SumSumgradients/relu_0_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџdd

Ћ
gradients/add_grad/Sum_1Sumgradients/relu_0_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:dd

g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
т
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџdd

п
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*&
_output_shapes
:dd


gradients/conv2d_0_grad/ShapeNShapeNConv_ReshapeVariable/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_0_grad/ConstConst*%
valueB"
   
      
   *
dtype0*
_output_shapes
:
с
+gradients/conv2d_0_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_0_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Н
,gradients/conv2d_0_grad/Conv2DBackpropFilterConv2DBackpropFilterConv_Reshapegradients/conv2d_0_grad/Const+gradients/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:




(gradients/conv2d_0_grad/tuple/group_depsNoOp,^gradients/conv2d_0_grad/Conv2DBackpropInput-^gradients/conv2d_0_grad/Conv2DBackpropFilter

0gradients/conv2d_0_grad/tuple/control_dependencyIdentity+gradients/conv2d_0_grad/Conv2DBackpropInput)^gradients/conv2d_0_grad/tuple/group_deps*>
_class4
20loc:@gradients/conv2d_0_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџdd*
T0

2gradients/conv2d_0_grad/tuple/control_dependency_1Identity,gradients/conv2d_0_grad/Conv2DBackpropFilter)^gradients/conv2d_0_grad/tuple/group_deps*?
_class5
31loc:@gradients/conv2d_0_grad/Conv2DBackpropFilter*&
_output_shapes
:


*
T0
{
beta1_power/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Variable*
dtype0

beta1_power
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape: *
dtype0*
_output_shapes
: 
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *wО?*
_class
loc:@Variable*
dtype0

beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@Variable
Ѕ
/Variable/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
      
   *
_class
loc:@Variable*
dtype0*
_output_shapes
:

%Variable/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable*
dtype0
п
Variable/Adam/Initializer/zerosFill/Variable/Adam/Initializer/zeros/shape_as_tensor%Variable/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable*&
_output_shapes
:



Ў
Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape:


*
dtype0*&
_output_shapes
:



Х
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:



{
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*&
_output_shapes
:



Ї
1Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
      
   *
_class
loc:@Variable*
dtype0*
_output_shapes
:

'Variable/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable*
dtype0*
_output_shapes
: 
х
!Variable/Adam_1/Initializer/zerosFill1Variable/Adam_1/Initializer/zeros/shape_as_tensor'Variable/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable*&
_output_shapes
:



А
Variable/Adam_1
VariableV2*
dtype0*&
_output_shapes
:


*
shared_name *
_class
loc:@Variable*
	container *
shape:



Ы
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:


*
use_locking(

Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*&
_output_shapes
:


*
T0
Љ
1Variable_1/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   d   d   
   *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:

'Variable_1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
ч
!Variable_1/Adam/Initializer/zerosFill1Variable_1/Adam/Initializer/zeros/shape_as_tensor'Variable_1/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_1*&
_output_shapes
:dd

В
Variable_1/Adam
VariableV2*
_class
loc:@Variable_1*
	container *
shape:dd
*
dtype0*&
_output_shapes
:dd
*
shared_name 
Э
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:dd
*
use_locking(

Variable_1/Adam/readIdentityVariable_1/Adam*&
_output_shapes
:dd
*
T0*
_class
loc:@Variable_1
Ћ
3Variable_1/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   d   d   
   *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:

)Variable_1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
э
#Variable_1/Adam_1/Initializer/zerosFill3Variable_1/Adam_1/Initializer/zeros/shape_as_tensor)Variable_1/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_1*&
_output_shapes
:dd

Д
Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
	container *
shape:dd
*
dtype0*&
_output_shapes
:dd

г
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:dd
*
use_locking(

Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*&
_output_shapes
:dd

Љ
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_2

'Variable_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
ч
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*&
_output_shapes
:




В
Variable_2/Adam
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape:



*
dtype0*&
_output_shapes
:




Э
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:





Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*&
_output_shapes
:




Ћ
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_2*
dtype0

)Variable_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
э
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*&
_output_shapes
:




Д
Variable_2/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_2*
	container *
shape:



*
dtype0*&
_output_shapes
:




г
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:



*
use_locking(

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*&
_output_shapes
:




Љ
1Variable_3/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

'Variable_3/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
ч
!Variable_3/Adam/Initializer/zerosFill1Variable_3/Adam/Initializer/zeros/shape_as_tensor'Variable_3/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_3*&
_output_shapes
:22

В
Variable_3/Adam
VariableV2*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_3*
	container *
shape:22
*
dtype0
Э
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:22


Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*&
_output_shapes
:22
*
T0
Ћ
3Variable_3/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

)Variable_3/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
э
#Variable_3/Adam_1/Initializer/zerosFill3Variable_3/Adam_1/Initializer/zeros/shape_as_tensor)Variable_3/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_3*&
_output_shapes
:22

Д
Variable_3/Adam_1
VariableV2*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_3*
	container *
shape:22

г
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:22


Variable_3/Adam_1/readIdentityVariable_3/Adam_1*&
_output_shapes
:22
*
T0*
_class
loc:@Variable_3
Љ
1Variable_4/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_4*
dtype0*
_output_shapes
:

'Variable_4/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
ч
!Variable_4/Adam/Initializer/zerosFill1Variable_4/Adam/Initializer/zeros/shape_as_tensor'Variable_4/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_4*&
_output_shapes
:




В
Variable_4/Adam
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_4*
	container *
shape:




Э
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_4

Variable_4/Adam/readIdentityVariable_4/Adam*&
_output_shapes
:



*
T0*
_class
loc:@Variable_4
Ћ
3Variable_4/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_4*
dtype0

)Variable_4/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
э
#Variable_4/Adam_1/Initializer/zerosFill3Variable_4/Adam_1/Initializer/zeros/shape_as_tensor)Variable_4/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_4*&
_output_shapes
:



*
T0
Д
Variable_4/Adam_1
VariableV2*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_4*
	container *
shape:



*
dtype0
г
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_4

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*&
_output_shapes
:




Љ
1Variable_5/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:

'Variable_5/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
ч
!Variable_5/Adam/Initializer/zerosFill1Variable_5/Adam/Initializer/zeros/shape_as_tensor'Variable_5/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_5*&
_output_shapes
:22
*
T0
В
Variable_5/Adam
VariableV2*
_class
loc:@Variable_5*
	container *
shape:22
*
dtype0*&
_output_shapes
:22
*
shared_name 
Э
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_5

Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*&
_output_shapes
:22

Ћ
3Variable_5/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:

)Variable_5/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
э
#Variable_5/Adam_1/Initializer/zerosFill3Variable_5/Adam_1/Initializer/zeros/shape_as_tensor)Variable_5/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_5*&
_output_shapes
:22

Д
Variable_5/Adam_1
VariableV2*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_5*
	container *
shape:22

г
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_5

Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*&
_output_shapes
:22

Љ
1Variable_6/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_6

'Variable_6/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_6*
dtype0*
_output_shapes
: 
ч
!Variable_6/Adam/Initializer/zerosFill1Variable_6/Adam/Initializer/zeros/shape_as_tensor'Variable_6/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_6*&
_output_shapes
:




В
Variable_6/Adam
VariableV2*
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_6*
	container 
Э
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_6

Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*&
_output_shapes
:




Ћ
3Variable_6/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_6

)Variable_6/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_6*
dtype0*
_output_shapes
: 
э
#Variable_6/Adam_1/Initializer/zerosFill3Variable_6/Adam_1/Initializer/zeros/shape_as_tensor)Variable_6/Adam_1/Initializer/zeros/Const*&
_output_shapes
:



*
T0*

index_type0*
_class
loc:@Variable_6
Д
Variable_6/Adam_1
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_6*
	container *
shape:




г
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
_class
loc:@Variable_6*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0

Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*&
_output_shapes
:




Љ
1Variable_7/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

'Variable_7/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
: 
ч
!Variable_7/Adam/Initializer/zerosFill1Variable_7/Adam/Initializer/zeros/shape_as_tensor'Variable_7/Adam/Initializer/zeros/Const*&
_output_shapes
:22
*
T0*

index_type0*
_class
loc:@Variable_7
В
Variable_7/Adam
VariableV2*
shape:22
*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_7*
	container 
Э
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_7*
validate_shape(*&
_output_shapes
:22
*
use_locking(

Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*&
_output_shapes
:22
*
T0
Ћ
3Variable_7/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

)Variable_7/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
: 
э
#Variable_7/Adam_1/Initializer/zerosFill3Variable_7/Adam_1/Initializer/zeros/shape_as_tensor)Variable_7/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_7*&
_output_shapes
:22

Д
Variable_7/Adam_1
VariableV2*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_7*
	container *
shape:22

г
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*&
_output_shapes
:22


Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7*&
_output_shapes
:22

Љ
1Variable_8/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_8*
dtype0*
_output_shapes
:

'Variable_8/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_8*
dtype0
ч
!Variable_8/Adam/Initializer/zerosFill1Variable_8/Adam/Initializer/zeros/shape_as_tensor'Variable_8/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_8*&
_output_shapes
:



*
T0
В
Variable_8/Adam
VariableV2*
_class
loc:@Variable_8*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name 
Э
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
_class
loc:@Variable_8*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0

Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_class
loc:@Variable_8*&
_output_shapes
:




Ћ
3Variable_8/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_8*
dtype0*
_output_shapes
:

)Variable_8/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_8*
dtype0
э
#Variable_8/Adam_1/Initializer/zerosFill3Variable_8/Adam_1/Initializer/zeros/shape_as_tensor)Variable_8/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_8*&
_output_shapes
:




Д
Variable_8/Adam_1
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_8*
	container *
shape:




г
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*&
_output_shapes
:





Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_class
loc:@Variable_8*&
_output_shapes
:




Љ
1Variable_9/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_9*
dtype0*
_output_shapes
:

'Variable_9/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
: 
ч
!Variable_9/Adam/Initializer/zerosFill1Variable_9/Adam/Initializer/zeros/shape_as_tensor'Variable_9/Adam/Initializer/zeros/Const*&
_output_shapes
:22
*
T0*

index_type0*
_class
loc:@Variable_9
В
Variable_9/Adam
VariableV2*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_9*
	container *
shape:22

Э
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*&
_output_shapes
:22


Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_class
loc:@Variable_9*&
_output_shapes
:22

Ћ
3Variable_9/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_9*
dtype0*
_output_shapes
:

)Variable_9/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_9*
dtype0
э
#Variable_9/Adam_1/Initializer/zerosFill3Variable_9/Adam_1/Initializer/zeros/shape_as_tensor)Variable_9/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_9*&
_output_shapes
:22

Д
Variable_9/Adam_1
VariableV2*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_9*
	container *
shape:22

г
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_9*
validate_shape(*&
_output_shapes
:22
*
use_locking(

Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_class
loc:@Variable_9*&
_output_shapes
:22

Ћ
2Variable_10/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_10*
dtype0*
_output_shapes
:

(Variable_10/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_10
ы
"Variable_10/Adam/Initializer/zerosFill2Variable_10/Adam/Initializer/zeros/shape_as_tensor(Variable_10/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_10*&
_output_shapes
:




Д
Variable_10/Adam
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_10*
	container *
shape:




б
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_10*
validate_shape(*&
_output_shapes
:



*
use_locking(

Variable_10/Adam/readIdentityVariable_10/Adam*
T0*
_class
loc:@Variable_10*&
_output_shapes
:




­
4Variable_10/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_10*
dtype0*
_output_shapes
:

*Variable_10/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_10*
dtype0*
_output_shapes
: 
ё
$Variable_10/Adam_1/Initializer/zerosFill4Variable_10/Adam_1/Initializer/zeros/shape_as_tensor*Variable_10/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_10*&
_output_shapes
:



*
T0
Ж
Variable_10/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_10*
	container *
shape:



*
dtype0*&
_output_shapes
:




з
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
_class
loc:@Variable_10*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0

Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*
_class
loc:@Variable_10*&
_output_shapes
:




Ћ
2Variable_11/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_11*
dtype0*
_output_shapes
:

(Variable_11/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_11*
dtype0*
_output_shapes
: 
ы
"Variable_11/Adam/Initializer/zerosFill2Variable_11/Adam/Initializer/zeros/shape_as_tensor(Variable_11/Adam/Initializer/zeros/Const*&
_output_shapes
:
*
T0*

index_type0*
_class
loc:@Variable_11
Д
Variable_11/Adam
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_11*
	container *
shape:

б
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
_class
loc:@Variable_11*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0

Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_class
loc:@Variable_11*&
_output_shapes
:

­
4Variable_11/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_11*
dtype0*
_output_shapes
:

*Variable_11/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_11*
dtype0
ё
$Variable_11/Adam_1/Initializer/zerosFill4Variable_11/Adam_1/Initializer/zeros/shape_as_tensor*Variable_11/Adam_1/Initializer/zeros/Const*&
_output_shapes
:
*
T0*

index_type0*
_class
loc:@Variable_11
Ж
Variable_11/Adam_1
VariableV2*
_class
loc:@Variable_11*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name 
з
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*&
_output_shapes
:


Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_class
loc:@Variable_11*&
_output_shapes
:

Ћ
2Variable_12/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_12*
dtype0*
_output_shapes
:

(Variable_12/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_12*
dtype0*
_output_shapes
: 
ы
"Variable_12/Adam/Initializer/zerosFill2Variable_12/Adam/Initializer/zeros/shape_as_tensor(Variable_12/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_12*&
_output_shapes
:




Д
Variable_12/Adam
VariableV2*
shared_name *
_class
loc:@Variable_12*
	container *
shape:



*
dtype0*&
_output_shapes
:




б
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
_class
loc:@Variable_12*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0

Variable_12/Adam/readIdentityVariable_12/Adam*
_class
loc:@Variable_12*&
_output_shapes
:



*
T0
­
4Variable_12/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_12*
dtype0*
_output_shapes
:

*Variable_12/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_12*
dtype0*
_output_shapes
: 
ё
$Variable_12/Adam_1/Initializer/zerosFill4Variable_12/Adam_1/Initializer/zeros/shape_as_tensor*Variable_12/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_12*&
_output_shapes
:




Ж
Variable_12/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_12*
	container *
shape:



*
dtype0*&
_output_shapes
:




з
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_12*
validate_shape(*&
_output_shapes
:



*
use_locking(

Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_class
loc:@Variable_12*&
_output_shapes
:




Ћ
2Variable_13/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_13*
dtype0*
_output_shapes
:

(Variable_13/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
: 
ы
"Variable_13/Adam/Initializer/zerosFill2Variable_13/Adam/Initializer/zeros/shape_as_tensor(Variable_13/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_13*&
_output_shapes
:
*
T0
Д
Variable_13/Adam
VariableV2*
shape:
*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_13*
	container 
б
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_13*
validate_shape(*&
_output_shapes
:
*
use_locking(

Variable_13/Adam/readIdentityVariable_13/Adam*&
_output_shapes
:
*
T0*
_class
loc:@Variable_13
­
4Variable_13/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_13*
dtype0*
_output_shapes
:

*Variable_13/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
: 
ё
$Variable_13/Adam_1/Initializer/zerosFill4Variable_13/Adam_1/Initializer/zeros/shape_as_tensor*Variable_13/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_13*&
_output_shapes
:

Ж
Variable_13/Adam_1
VariableV2*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_13
з
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*&
_output_shapes
:


Variable_13/Adam_1/readIdentityVariable_13/Adam_1*
T0*
_class
loc:@Variable_13*&
_output_shapes
:

Ћ
2Variable_14/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_14

(Variable_14/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_14*
dtype0*
_output_shapes
: 
ы
"Variable_14/Adam/Initializer/zerosFill2Variable_14/Adam/Initializer/zeros/shape_as_tensor(Variable_14/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_14*&
_output_shapes
:




Д
Variable_14/Adam
VariableV2*
shared_name *
_class
loc:@Variable_14*
	container *
shape:



*
dtype0*&
_output_shapes
:




б
Variable_14/Adam/AssignAssignVariable_14/Adam"Variable_14/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(*&
_output_shapes
:





Variable_14/Adam/readIdentityVariable_14/Adam*
T0*
_class
loc:@Variable_14*&
_output_shapes
:




­
4Variable_14/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_14*
dtype0*
_output_shapes
:

*Variable_14/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_14*
dtype0
ё
$Variable_14/Adam_1/Initializer/zerosFill4Variable_14/Adam_1/Initializer/zeros/shape_as_tensor*Variable_14/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_14*&
_output_shapes
:




Ж
Variable_14/Adam_1
VariableV2*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_14*
	container *
shape:



*
dtype0
з
Variable_14/Adam_1/AssignAssignVariable_14/Adam_1$Variable_14/Adam_1/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(

Variable_14/Adam_1/readIdentityVariable_14/Adam_1*
T0*
_class
loc:@Variable_14*&
_output_shapes
:




Ћ
2Variable_15/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

(Variable_15/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_15
ы
"Variable_15/Adam/Initializer/zerosFill2Variable_15/Adam/Initializer/zeros/shape_as_tensor(Variable_15/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_15*&
_output_shapes
:

Д
Variable_15/Adam
VariableV2*
shared_name *
_class
loc:@Variable_15*
	container *
shape:
*
dtype0*&
_output_shapes
:

б
Variable_15/Adam/AssignAssignVariable_15/Adam"Variable_15/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*&
_output_shapes
:


Variable_15/Adam/readIdentityVariable_15/Adam*&
_output_shapes
:
*
T0*
_class
loc:@Variable_15
­
4Variable_15/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

*Variable_15/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
: 
ё
$Variable_15/Adam_1/Initializer/zerosFill4Variable_15/Adam_1/Initializer/zeros/shape_as_tensor*Variable_15/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_15*&
_output_shapes
:
*
T0
Ж
Variable_15/Adam_1
VariableV2*
_class
loc:@Variable_15*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name 
з
Variable_15/Adam_1/AssignAssignVariable_15/Adam_1$Variable_15/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*&
_output_shapes
:


Variable_15/Adam_1/readIdentityVariable_15/Adam_1*
_class
loc:@Variable_15*&
_output_shapes
:
*
T0
Ћ
2Variable_16/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_16*
dtype0*
_output_shapes
:

(Variable_16/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_16*
dtype0*
_output_shapes
: 
ы
"Variable_16/Adam/Initializer/zerosFill2Variable_16/Adam/Initializer/zeros/shape_as_tensor(Variable_16/Adam/Initializer/zeros/Const*&
_output_shapes
:



*
T0*

index_type0*
_class
loc:@Variable_16
Д
Variable_16/Adam
VariableV2*
shared_name *
_class
loc:@Variable_16*
	container *
shape:



*
dtype0*&
_output_shapes
:




б
Variable_16/Adam/AssignAssignVariable_16/Adam"Variable_16/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_16

Variable_16/Adam/readIdentityVariable_16/Adam*
T0*
_class
loc:@Variable_16*&
_output_shapes
:




­
4Variable_16/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_16*
dtype0

*Variable_16/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_16
ё
$Variable_16/Adam_1/Initializer/zerosFill4Variable_16/Adam_1/Initializer/zeros/shape_as_tensor*Variable_16/Adam_1/Initializer/zeros/Const*&
_output_shapes
:



*
T0*

index_type0*
_class
loc:@Variable_16
Ж
Variable_16/Adam_1
VariableV2*
_class
loc:@Variable_16*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name 
з
Variable_16/Adam_1/AssignAssignVariable_16/Adam_1$Variable_16/Adam_1/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(

Variable_16/Adam_1/readIdentityVariable_16/Adam_1*&
_output_shapes
:



*
T0*
_class
loc:@Variable_16
Ћ
2Variable_17/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_17*
dtype0*
_output_shapes
:

(Variable_17/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_17*
dtype0*
_output_shapes
: 
ы
"Variable_17/Adam/Initializer/zerosFill2Variable_17/Adam/Initializer/zeros/shape_as_tensor(Variable_17/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_17*&
_output_shapes
:

Д
Variable_17/Adam
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_17*
	container *
shape:

б
Variable_17/Adam/AssignAssignVariable_17/Adam"Variable_17/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*&
_output_shapes
:


Variable_17/Adam/readIdentityVariable_17/Adam*&
_output_shapes
:
*
T0*
_class
loc:@Variable_17
­
4Variable_17/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_17*
dtype0*
_output_shapes
:

*Variable_17/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_17*
dtype0*
_output_shapes
: 
ё
$Variable_17/Adam_1/Initializer/zerosFill4Variable_17/Adam_1/Initializer/zeros/shape_as_tensor*Variable_17/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_17*&
_output_shapes
:

Ж
Variable_17/Adam_1
VariableV2*
_class
loc:@Variable_17*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name 
з
Variable_17/Adam_1/AssignAssignVariable_17/Adam_1$Variable_17/Adam_1/Initializer/zeros*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(

Variable_17/Adam_1/readIdentityVariable_17/Adam_1*
T0*
_class
loc:@Variable_17*&
_output_shapes
:

Ћ
2Variable_18/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_18*
dtype0

(Variable_18/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_18*
dtype0*
_output_shapes
: 
ы
"Variable_18/Adam/Initializer/zerosFill2Variable_18/Adam/Initializer/zeros/shape_as_tensor(Variable_18/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_18*&
_output_shapes
:




Д
Variable_18/Adam
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_18*
	container *
shape:




б
Variable_18/Adam/AssignAssignVariable_18/Adam"Variable_18/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:





Variable_18/Adam/readIdentityVariable_18/Adam*
T0*
_class
loc:@Variable_18*&
_output_shapes
:




­
4Variable_18/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_18*
dtype0*
_output_shapes
:

*Variable_18/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_18*
dtype0*
_output_shapes
: 
ё
$Variable_18/Adam_1/Initializer/zerosFill4Variable_18/Adam_1/Initializer/zeros/shape_as_tensor*Variable_18/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_18*&
_output_shapes
:




Ж
Variable_18/Adam_1
VariableV2*
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_18*
	container 
з
Variable_18/Adam_1/AssignAssignVariable_18/Adam_1$Variable_18/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:



*
use_locking(

Variable_18/Adam_1/readIdentityVariable_18/Adam_1*
T0*
_class
loc:@Variable_18*&
_output_shapes
:




Ћ
2Variable_19/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         
   *
_class
loc:@Variable_19

(Variable_19/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_19*
dtype0*
_output_shapes
: 
ы
"Variable_19/Adam/Initializer/zerosFill2Variable_19/Adam/Initializer/zeros/shape_as_tensor(Variable_19/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_19*&
_output_shapes
:

Д
Variable_19/Adam
VariableV2*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_19
б
Variable_19/Adam/AssignAssignVariable_19/Adam"Variable_19/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*&
_output_shapes
:


Variable_19/Adam/readIdentityVariable_19/Adam*
T0*
_class
loc:@Variable_19*&
_output_shapes
:

­
4Variable_19/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_19*
dtype0*
_output_shapes
:

*Variable_19/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_19*
dtype0*
_output_shapes
: 
ё
$Variable_19/Adam_1/Initializer/zerosFill4Variable_19/Adam_1/Initializer/zeros/shape_as_tensor*Variable_19/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_19*&
_output_shapes
:

Ж
Variable_19/Adam_1
VariableV2*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_19
з
Variable_19/Adam_1/AssignAssignVariable_19/Adam_1$Variable_19/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*&
_output_shapes
:


Variable_19/Adam_1/readIdentityVariable_19/Adam_1*
T0*
_class
loc:@Variable_19*&
_output_shapes
:

Ћ
2Variable_20/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_20*
dtype0*
_output_shapes
:

(Variable_20/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 
ы
"Variable_20/Adam/Initializer/zerosFill2Variable_20/Adam/Initializer/zeros/shape_as_tensor(Variable_20/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_20*&
_output_shapes
:




Д
Variable_20/Adam
VariableV2*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_20*
	container *
shape:



*
dtype0
б
Variable_20/Adam/AssignAssignVariable_20/Adam"Variable_20/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_20*
validate_shape(*&
_output_shapes
:



*
use_locking(

Variable_20/Adam/readIdentityVariable_20/Adam*&
_output_shapes
:



*
T0*
_class
loc:@Variable_20
­
4Variable_20/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_20*
dtype0*
_output_shapes
:

*Variable_20/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 
ё
$Variable_20/Adam_1/Initializer/zerosFill4Variable_20/Adam_1/Initializer/zeros/shape_as_tensor*Variable_20/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_20*&
_output_shapes
:



*
T0
Ж
Variable_20/Adam_1
VariableV2*
_class
loc:@Variable_20*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name 
з
Variable_20/Adam_1/AssignAssignVariable_20/Adam_1$Variable_20/Adam_1/Initializer/zeros*
_class
loc:@Variable_20*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0

Variable_20/Adam_1/readIdentityVariable_20/Adam_1*
T0*
_class
loc:@Variable_20*&
_output_shapes
:




Ћ
2Variable_21/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_21*
dtype0*
_output_shapes
:

(Variable_21/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 
ы
"Variable_21/Adam/Initializer/zerosFill2Variable_21/Adam/Initializer/zeros/shape_as_tensor(Variable_21/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_21*&
_output_shapes
:

Д
Variable_21/Adam
VariableV2*
_class
loc:@Variable_21*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name 
б
Variable_21/Adam/AssignAssignVariable_21/Adam"Variable_21/Adam/Initializer/zeros*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(

Variable_21/Adam/readIdentityVariable_21/Adam*
_class
loc:@Variable_21*&
_output_shapes
:
*
T0
­
4Variable_21/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_21*
dtype0*
_output_shapes
:

*Variable_21/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_21
ё
$Variable_21/Adam_1/Initializer/zerosFill4Variable_21/Adam_1/Initializer/zeros/shape_as_tensor*Variable_21/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_21*&
_output_shapes
:

Ж
Variable_21/Adam_1
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_21*
	container *
shape:

з
Variable_21/Adam_1/AssignAssignVariable_21/Adam_1$Variable_21/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_21*
validate_shape(*&
_output_shapes
:
*
use_locking(

Variable_21/Adam_1/readIdentityVariable_21/Adam_1*
T0*
_class
loc:@Variable_21*&
_output_shapes
:

Ћ
2Variable_22/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_22*
dtype0*
_output_shapes
:

(Variable_22/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_22*
dtype0*
_output_shapes
: 
ы
"Variable_22/Adam/Initializer/zerosFill2Variable_22/Adam/Initializer/zeros/shape_as_tensor(Variable_22/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_22*&
_output_shapes
:



*
T0
Д
Variable_22/Adam
VariableV2*
shared_name *
_class
loc:@Variable_22*
	container *
shape:



*
dtype0*&
_output_shapes
:




б
Variable_22/Adam/AssignAssignVariable_22/Adam"Variable_22/Adam/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(

Variable_22/Adam/readIdentityVariable_22/Adam*
T0*
_class
loc:@Variable_22*&
_output_shapes
:




­
4Variable_22/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_22*
dtype0*
_output_shapes
:

*Variable_22/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_22*
dtype0*
_output_shapes
: 
ё
$Variable_22/Adam_1/Initializer/zerosFill4Variable_22/Adam_1/Initializer/zeros/shape_as_tensor*Variable_22/Adam_1/Initializer/zeros/Const*&
_output_shapes
:



*
T0*

index_type0*
_class
loc:@Variable_22
Ж
Variable_22/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_22*
	container *
shape:



*
dtype0*&
_output_shapes
:




з
Variable_22/Adam_1/AssignAssignVariable_22/Adam_1$Variable_22/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
:





Variable_22/Adam_1/readIdentityVariable_22/Adam_1*
T0*
_class
loc:@Variable_22*&
_output_shapes
:




Ћ
2Variable_23/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         
   *
_class
loc:@Variable_23

(Variable_23/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_23*
dtype0*
_output_shapes
: 
ы
"Variable_23/Adam/Initializer/zerosFill2Variable_23/Adam/Initializer/zeros/shape_as_tensor(Variable_23/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_23*&
_output_shapes
:

Д
Variable_23/Adam
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_23*
	container *
shape:

б
Variable_23/Adam/AssignAssignVariable_23/Adam"Variable_23/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_23*
validate_shape(*&
_output_shapes
:


Variable_23/Adam/readIdentityVariable_23/Adam*
T0*
_class
loc:@Variable_23*&
_output_shapes
:

­
4Variable_23/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_23*
dtype0*
_output_shapes
:

*Variable_23/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_23*
dtype0*
_output_shapes
: 
ё
$Variable_23/Adam_1/Initializer/zerosFill4Variable_23/Adam_1/Initializer/zeros/shape_as_tensor*Variable_23/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_23*&
_output_shapes
:

Ж
Variable_23/Adam_1
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_23*
	container *
shape:

з
Variable_23/Adam_1/AssignAssignVariable_23/Adam_1$Variable_23/Adam_1/Initializer/zeros*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_23*
validate_shape(

Variable_23/Adam_1/readIdentityVariable_23/Adam_1*
T0*
_class
loc:@Variable_23*&
_output_shapes
:

Ћ
2Variable_24/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_24*
dtype0*
_output_shapes
:

(Variable_24/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_24*
dtype0*
_output_shapes
: 
ы
"Variable_24/Adam/Initializer/zerosFill2Variable_24/Adam/Initializer/zeros/shape_as_tensor(Variable_24/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_24*&
_output_shapes
:



*
T0
Д
Variable_24/Adam
VariableV2*
shared_name *
_class
loc:@Variable_24*
	container *
shape:



*
dtype0*&
_output_shapes
:




б
Variable_24/Adam/AssignAssignVariable_24/Adam"Variable_24/Adam/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(

Variable_24/Adam/readIdentityVariable_24/Adam*&
_output_shapes
:



*
T0*
_class
loc:@Variable_24
­
4Variable_24/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_24*
dtype0*
_output_shapes
:

*Variable_24/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_24*
dtype0*
_output_shapes
: 
ё
$Variable_24/Adam_1/Initializer/zerosFill4Variable_24/Adam_1/Initializer/zeros/shape_as_tensor*Variable_24/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_24*&
_output_shapes
:




Ж
Variable_24/Adam_1
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_24*
	container *
shape:




з
Variable_24/Adam_1/AssignAssignVariable_24/Adam_1$Variable_24/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(*&
_output_shapes
:





Variable_24/Adam_1/readIdentityVariable_24/Adam_1*
T0*
_class
loc:@Variable_24*&
_output_shapes
:




Ћ
2Variable_25/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_25*
dtype0*
_output_shapes
:

(Variable_25/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_25*
dtype0*
_output_shapes
: 
ы
"Variable_25/Adam/Initializer/zerosFill2Variable_25/Adam/Initializer/zeros/shape_as_tensor(Variable_25/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_25*&
_output_shapes
:

Д
Variable_25/Adam
VariableV2*
_class
loc:@Variable_25*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name 
б
Variable_25/Adam/AssignAssignVariable_25/Adam"Variable_25/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(*&
_output_shapes
:


Variable_25/Adam/readIdentityVariable_25/Adam*
T0*
_class
loc:@Variable_25*&
_output_shapes
:

­
4Variable_25/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_25*
dtype0*
_output_shapes
:

*Variable_25/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_25*
dtype0*
_output_shapes
: 
ё
$Variable_25/Adam_1/Initializer/zerosFill4Variable_25/Adam_1/Initializer/zeros/shape_as_tensor*Variable_25/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_25*&
_output_shapes
:

Ж
Variable_25/Adam_1
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_25*
	container *
shape:

з
Variable_25/Adam_1/AssignAssignVariable_25/Adam_1$Variable_25/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(*&
_output_shapes
:


Variable_25/Adam_1/readIdentityVariable_25/Adam_1*&
_output_shapes
:
*
T0*
_class
loc:@Variable_25
Ѓ
2Variable_26/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"ъ  	   *
_class
loc:@Variable_26*
dtype0*
_output_shapes
:

(Variable_26/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_26
ф
"Variable_26/Adam/Initializer/zerosFill2Variable_26/Adam/Initializer/zeros/shape_as_tensor(Variable_26/Adam/Initializer/zeros/Const*
_output_shapes
:	ъ	*
T0*

index_type0*
_class
loc:@Variable_26
І
Variable_26/Adam
VariableV2*
_class
loc:@Variable_26*
	container *
shape:	ъ	*
dtype0*
_output_shapes
:	ъ	*
shared_name 
Ъ
Variable_26/Adam/AssignAssignVariable_26/Adam"Variable_26/Adam/Initializer/zeros*
_class
loc:@Variable_26*
validate_shape(*
_output_shapes
:	ъ	*
use_locking(*
T0
}
Variable_26/Adam/readIdentityVariable_26/Adam*
_class
loc:@Variable_26*
_output_shapes
:	ъ	*
T0
Ѕ
4Variable_26/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"ъ  	   *
_class
loc:@Variable_26*
dtype0*
_output_shapes
:

*Variable_26/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_26
ъ
$Variable_26/Adam_1/Initializer/zerosFill4Variable_26/Adam_1/Initializer/zeros/shape_as_tensor*Variable_26/Adam_1/Initializer/zeros/Const*
_output_shapes
:	ъ	*
T0*

index_type0*
_class
loc:@Variable_26
Ј
Variable_26/Adam_1
VariableV2*
_output_shapes
:	ъ	*
shared_name *
_class
loc:@Variable_26*
	container *
shape:	ъ	*
dtype0
а
Variable_26/Adam_1/AssignAssignVariable_26/Adam_1$Variable_26/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_26*
validate_shape(*
_output_shapes
:	ъ	

Variable_26/Adam_1/readIdentityVariable_26/Adam_1*
_output_shapes
:	ъ	*
T0*
_class
loc:@Variable_26
Ѓ
2Variable_27/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   	   *
_class
loc:@Variable_27*
dtype0*
_output_shapes
:

(Variable_27/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_27
у
"Variable_27/Adam/Initializer/zerosFill2Variable_27/Adam/Initializer/zeros/shape_as_tensor(Variable_27/Adam/Initializer/zeros/Const*
_output_shapes

:	*
T0*

index_type0*
_class
loc:@Variable_27
Є
Variable_27/Adam
VariableV2*
	container *
shape
:	*
dtype0*
_output_shapes

:	*
shared_name *
_class
loc:@Variable_27
Щ
Variable_27/Adam/AssignAssignVariable_27/Adam"Variable_27/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:	*
use_locking(*
T0*
_class
loc:@Variable_27
|
Variable_27/Adam/readIdentityVariable_27/Adam*
_output_shapes

:	*
T0*
_class
loc:@Variable_27
Ѕ
4Variable_27/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   	   *
_class
loc:@Variable_27*
dtype0*
_output_shapes
:

*Variable_27/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_27*
dtype0
щ
$Variable_27/Adam_1/Initializer/zerosFill4Variable_27/Adam_1/Initializer/zeros/shape_as_tensor*Variable_27/Adam_1/Initializer/zeros/Const*
_output_shapes

:	*
T0*

index_type0*
_class
loc:@Variable_27
І
Variable_27/Adam_1
VariableV2*
_class
loc:@Variable_27*
	container *
shape
:	*
dtype0*
_output_shapes

:	*
shared_name 
Я
Variable_27/Adam_1/AssignAssignVariable_27/Adam_1$Variable_27/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:	*
use_locking(*
T0*
_class
loc:@Variable_27

Variable_27/Adam_1/readIdentityVariable_27/Adam_1*
T0*
_class
loc:@Variable_27*
_output_shapes

:	
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *Зб8*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
м
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_0_grad/tuple/control_dependency_1*&
_output_shapes
:


*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( 
с
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *&
_output_shapes
:dd

ц
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( *&
_output_shapes
:




у
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_3*
use_nesterov( *&
_output_shapes
:22
*
use_locking( *
T0
ц
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_2_grad/tuple/control_dependency_1*
_class
loc:@Variable_4*
use_nesterov( *&
_output_shapes
:



*
use_locking( *
T0
у
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*&
_output_shapes
:22
*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( 
ц
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*
use_nesterov( *&
_output_shapes
:




у
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
_class
loc:@Variable_7*
use_nesterov( *&
_output_shapes
:22
*
use_locking( *
T0
ц
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_4_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_8*
use_nesterov( *&
_output_shapes
:



*
use_locking( 
у
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_9*
use_nesterov( *&
_output_shapes
:22

ы
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_5_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_10*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_5_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_11*
use_nesterov( *&
_output_shapes
:
*
use_locking( 
ы
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_6_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_12*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_6_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_13*
use_nesterov( *&
_output_shapes
:

ы
!Adam/update_Variable_14/ApplyAdam	ApplyAdamVariable_14Variable_14/AdamVariable_14/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_7_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_14*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_15/ApplyAdam	ApplyAdamVariable_15Variable_15/AdamVariable_15/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_7_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_15*
use_nesterov( *&
_output_shapes
:
*
use_locking( 
ы
!Adam/update_Variable_16/ApplyAdam	ApplyAdamVariable_16Variable_16/AdamVariable_16/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_8_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:



*
use_locking( *
T0*
_class
loc:@Variable_16
ш
!Adam/update_Variable_17/ApplyAdam	ApplyAdamVariable_17Variable_17/AdamVariable_17/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_8_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_17*
use_nesterov( *&
_output_shapes
:

ы
!Adam/update_Variable_18/ApplyAdam	ApplyAdamVariable_18Variable_18/AdamVariable_18/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_9_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_18*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_19/ApplyAdam	ApplyAdamVariable_19Variable_19/AdamVariable_19/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_9_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_19*
use_nesterov( *&
_output_shapes
:

ь
!Adam/update_Variable_20/ApplyAdam	ApplyAdamVariable_20Variable_20/AdamVariable_20/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/conv2d_10_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_20*
use_nesterov( *&
_output_shapes
:



*
use_locking( 
щ
!Adam/update_Variable_21/ApplyAdam	ApplyAdamVariable_21Variable_21/AdamVariable_21/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_10_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_21*
use_nesterov( *&
_output_shapes
:
*
use_locking( 
ь
!Adam/update_Variable_22/ApplyAdam	ApplyAdamVariable_22Variable_22/AdamVariable_22/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/conv2d_11_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_22*
use_nesterov( *&
_output_shapes
:



*
use_locking( 
щ
!Adam/update_Variable_23/ApplyAdam	ApplyAdamVariable_23Variable_23/AdamVariable_23/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_11_grad/tuple/control_dependency_1*
_class
loc:@Variable_23*
use_nesterov( *&
_output_shapes
:
*
use_locking( *
T0
ь
!Adam/update_Variable_24/ApplyAdam	ApplyAdamVariable_24Variable_24/AdamVariable_24/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/conv2d_12_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_24*
use_nesterov( *&
_output_shapes
:



*
use_locking( 
щ
!Adam/update_Variable_25/ApplyAdam	ApplyAdamVariable_25Variable_25/AdamVariable_25/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_12_grad/tuple/control_dependency_1*&
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@Variable_25*
use_nesterov( 
х
!Adam/update_Variable_26/ApplyAdam	ApplyAdamVariable_26Variable_26/AdamVariable_26/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/FC_MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_26*
use_nesterov( *
_output_shapes
:	ъ	*
use_locking( 
с
!Adam/update_Variable_27/ApplyAdam	ApplyAdamVariable_27Variable_27/AdamVariable_27/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_13_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_27*
use_nesterov( *
_output_shapes

:	
Я
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam"^Adam/update_Variable_26/ApplyAdam"^Adam/update_Variable_27/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@Variable
б

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam"^Adam/update_Variable_26/ApplyAdam"^Adam/update_Variable_27/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 

AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam"^Adam/update_Variable_26/ApplyAdam"^Adam/update_Variable_27/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Т
save/SaveV2/tensor_namesConst*ѕ
valueыBшBVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15BVariable_16BVariable_17BVariable_18BVariable_19B
Variable_2BVariable_20BVariable_21BVariable_22BVariable_23BVariable_24BVariable_25BVariable_26BVariable_27B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
Variable_8B
Variable_9*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
х
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1Variable_10Variable_11Variable_12Variable_13Variable_14Variable_15Variable_16Variable_17Variable_18Variable_19
Variable_2Variable_20Variable_21Variable_22Variable_23Variable_24Variable_25Variable_26Variable_27
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7
Variable_8
Variable_9**
dtypes 
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
_output_shapes
: *
T0
д
save/RestoreV2/tensor_namesConst"/device:CPU:0*ѕ
valueыBшBVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15BVariable_16BVariable_17BVariable_18BVariable_19B
Variable_2BVariable_20BVariable_21BVariable_22BVariable_23BVariable_24BVariable_25BVariable_26BVariable_27B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
Variable_8B
Variable_9*
dtype0*
_output_shapes
:
­
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ї
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0**
dtypes 
2*
_output_shapesr
p::::::::::::::::::::::::::::
І
save/AssignAssignVariablesave/RestoreV2*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:


*
use_locking(
Ў
save/Assign_1Assign
Variable_1save/RestoreV2:1*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:dd
*
use_locking(
А
save/Assign_2AssignVariable_10save/RestoreV2:2*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*&
_output_shapes
:




А
save/Assign_3AssignVariable_11save/RestoreV2:3*
T0*
_class
loc:@Variable_11*
validate_shape(*&
_output_shapes
:
*
use_locking(
А
save/Assign_4AssignVariable_12save/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*&
_output_shapes
:




А
save/Assign_5AssignVariable_13save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*&
_output_shapes
:

А
save/Assign_6AssignVariable_14save/RestoreV2:6*
T0*
_class
loc:@Variable_14*
validate_shape(*&
_output_shapes
:



*
use_locking(
А
save/Assign_7AssignVariable_15save/RestoreV2:7*
_class
loc:@Variable_15*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0
А
save/Assign_8AssignVariable_16save/RestoreV2:8*
_class
loc:@Variable_16*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
А
save/Assign_9AssignVariable_17save/RestoreV2:9*
T0*
_class
loc:@Variable_17*
validate_shape(*&
_output_shapes
:
*
use_locking(
В
save/Assign_10AssignVariable_18save/RestoreV2:10*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:



*
use_locking(
В
save/Assign_11AssignVariable_19save/RestoreV2:11*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_19
А
save/Assign_12Assign
Variable_2save/RestoreV2:12*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:




В
save/Assign_13AssignVariable_20save/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*&
_output_shapes
:




В
save/Assign_14AssignVariable_21save/RestoreV2:14*
T0*
_class
loc:@Variable_21*
validate_shape(*&
_output_shapes
:
*
use_locking(
В
save/Assign_15AssignVariable_22save/RestoreV2:15*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
В
save/Assign_16AssignVariable_23save/RestoreV2:16*
_class
loc:@Variable_23*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0
В
save/Assign_17AssignVariable_24save/RestoreV2:17*
T0*
_class
loc:@Variable_24*
validate_shape(*&
_output_shapes
:



*
use_locking(
В
save/Assign_18AssignVariable_25save/RestoreV2:18*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_25
Ћ
save/Assign_19AssignVariable_26save/RestoreV2:19*
T0*
_class
loc:@Variable_26*
validate_shape(*
_output_shapes
:	ъ	*
use_locking(
Њ
save/Assign_20AssignVariable_27save/RestoreV2:20*
validate_shape(*
_output_shapes

:	*
use_locking(*
T0*
_class
loc:@Variable_27
А
save/Assign_21Assign
Variable_3save/RestoreV2:21*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(
А
save/Assign_22Assign
Variable_4save/RestoreV2:22*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
А
save/Assign_23Assign
Variable_5save/RestoreV2:23*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
:22
*
use_locking(
А
save/Assign_24Assign
Variable_6save/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*&
_output_shapes
:




А
save/Assign_25Assign
Variable_7save/RestoreV2:25*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*&
_output_shapes
:22

А
save/Assign_26Assign
Variable_8save/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*&
_output_shapes
:




А
save/Assign_27Assign
Variable_9save/RestoreV2:27*
T0*
_class
loc:@Variable_9*
validate_shape(*&
_output_shapes
:22
*
use_locking(
ш
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27"!Pє     л}	ЗcріЖжAJїш
В$$
:
Add
x"T
y"T
z"T"
Ttype:
2	
ю
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ы
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisintџџџџџџџџџ"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.7.02v1.7.0-3-g024aecf414мђ

R
Placeholder_xPlaceholder*
_output_shapes
:*
shape:*
dtype0
R
Placeholder_yPlaceholder*
shape:*
dtype0*
_output_shapes
:
k
Conv_Reshape/shapeConst*%
valueB"џџџџd   d      *
dtype0*
_output_shapes
:

Conv_ReshapeReshapePlaceholder_xConv_Reshape/shape*/
_output_shapes
:џџџџџџџџџdd*
T0*
Tshape0
g
Kernel_0/shapeConst*%
valueB"
   
      
   *
dtype0*
_output_shapes
:
R
Kernel_0/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_0/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_0/RandomStandardNormalRandomStandardNormalKernel_0/shape*
dtype0*&
_output_shapes
:


*
seed2 *

seed *
T0
t
Kernel_0/mulMulKernel_0/RandomStandardNormalKernel_0/stddev*&
_output_shapes
:


*
T0
]
Kernel_0AddKernel_0/mulKernel_0/mean*
T0*&
_output_shapes
:




Variable
VariableV2*
dtype0*&
_output_shapes
:


*
	container *
shape:


*
shared_name 
Є
Variable/AssignAssignVariableKernel_0*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:


*
use_locking(
q
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*&
_output_shapes
:



з
conv2d_0Conv2DConv_ReshapeVariable/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџdd
*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
l
Kernel_Bias_0/shapeConst*
_output_shapes
:*%
valueB"   d   d   
   *
dtype0
W
Kernel_Bias_0/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_0/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_0/RandomStandardNormalRandomStandardNormalKernel_Bias_0/shape*

seed *
T0*
dtype0*&
_output_shapes
:dd
*
seed2 

Kernel_Bias_0/mulMul"Kernel_Bias_0/RandomStandardNormalKernel_Bias_0/stddev*
T0*&
_output_shapes
:dd

l
Kernel_Bias_0AddKernel_Bias_0/mulKernel_Bias_0/mean*
T0*&
_output_shapes
:dd



Variable_1
VariableV2*
dtype0*&
_output_shapes
:dd
*
	container *
shape:dd
*
shared_name 
Џ
Variable_1/AssignAssign
Variable_1Kernel_Bias_0*
use_locking(*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:dd

w
Variable_1/readIdentity
Variable_1*&
_output_shapes
:dd
*
T0*
_class
loc:@Variable_1
_
addAddconv2d_0Variable_1/read*
T0*/
_output_shapes
:џџџџџџџџџdd

M
relu_0Reluadd*/
_output_shapes
:џџџџџџџџџdd
*
T0
Ѕ
Pool_0MaxPoolrelu_0*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22

g
Kernel_1/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
T
Kernel_1/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_1/RandomStandardNormalRandomStandardNormalKernel_1/shape*&
_output_shapes
:



*
seed2 *

seed *
T0*
dtype0
t
Kernel_1/mulMulKernel_1/RandomStandardNormalKernel_1/stddev*
T0*&
_output_shapes
:




]
Kernel_1AddKernel_1/mulKernel_1/mean*&
_output_shapes
:



*
T0


Variable_2
VariableV2*
shape:



*
shared_name *
dtype0*&
_output_shapes
:



*
	container 
Њ
Variable_2/AssignAssign
Variable_2Kernel_1*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_2
w
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*&
_output_shapes
:




г
conv2d_1Conv2DPool_0Variable_2/read*/
_output_shapes
:џџџџџџџџџ22
*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
l
Kernel_Bias_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   2   2   
   
W
Kernel_Bias_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_1/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_1/RandomStandardNormalRandomStandardNormalKernel_Bias_1/shape*&
_output_shapes
:22
*
seed2 *

seed *
T0*
dtype0

Kernel_Bias_1/mulMul"Kernel_Bias_1/RandomStandardNormalKernel_Bias_1/stddev*
T0*&
_output_shapes
:22

l
Kernel_Bias_1AddKernel_Bias_1/mulKernel_Bias_1/mean*
T0*&
_output_shapes
:22



Variable_3
VariableV2*
dtype0*&
_output_shapes
:22
*
	container *
shape:22
*
shared_name 
Џ
Variable_3/AssignAssign
Variable_3Kernel_Bias_1*
validate_shape(*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_3
w
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*&
_output_shapes
:22
*
T0
a
add_1Addconv2d_1Variable_3/read*/
_output_shapes
:џџџџџџџџџ22
*
T0
O
relu_1Reluadd_1*/
_output_shapes
:џџџџџџџџџ22
*
T0
g
Kernel_2/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
T
Kernel_2/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_2/RandomStandardNormalRandomStandardNormalKernel_2/shape*
dtype0*&
_output_shapes
:



*
seed2 *

seed *
T0
t
Kernel_2/mulMulKernel_2/RandomStandardNormalKernel_2/stddev*
T0*&
_output_shapes
:




]
Kernel_2AddKernel_2/mulKernel_2/mean*
T0*&
_output_shapes
:






Variable_4
VariableV2*
shared_name *
dtype0*&
_output_shapes
:



*
	container *
shape:




Њ
Variable_4/AssignAssign
Variable_4Kernel_2*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(
w
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*&
_output_shapes
:



*
T0
г
conv2d_2Conv2Drelu_1Variable_4/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22

l
Kernel_Bias_2/shapeConst*
_output_shapes
:*%
valueB"   2   2   
   *
dtype0
W
Kernel_Bias_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_2/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_2/RandomStandardNormalRandomStandardNormalKernel_Bias_2/shape*
T0*
dtype0*&
_output_shapes
:22
*
seed2 *

seed 

Kernel_Bias_2/mulMul"Kernel_Bias_2/RandomStandardNormalKernel_Bias_2/stddev*
T0*&
_output_shapes
:22

l
Kernel_Bias_2AddKernel_Bias_2/mulKernel_Bias_2/mean*&
_output_shapes
:22
*
T0


Variable_5
VariableV2*&
_output_shapes
:22
*
	container *
shape:22
*
shared_name *
dtype0
Џ
Variable_5/AssignAssign
Variable_5Kernel_Bias_2*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
:22

w
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*&
_output_shapes
:22
*
T0
a
add_2Addconv2d_2Variable_5/read*
T0*/
_output_shapes
:џџџџџџџџџ22

O
relu_2Reluadd_2*/
_output_shapes
:џџџџџџџџџ22
*
T0
g
Kernel_3/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_3/stddevConst*
_output_shapes
: *
valueB
 *ЭЬL=*
dtype0

Kernel_3/RandomStandardNormalRandomStandardNormalKernel_3/shape*

seed *
T0*
dtype0*&
_output_shapes
:



*
seed2 
t
Kernel_3/mulMulKernel_3/RandomStandardNormalKernel_3/stddev*&
_output_shapes
:



*
T0
]
Kernel_3AddKernel_3/mulKernel_3/mean*&
_output_shapes
:



*
T0


Variable_6
VariableV2*
shared_name *
dtype0*&
_output_shapes
:



*
	container *
shape:




Њ
Variable_6/AssignAssign
Variable_6Kernel_3*
_class
loc:@Variable_6*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
w
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*&
_output_shapes
:




г
conv2d_3Conv2Drelu_2Variable_6/read*/
_output_shapes
:џџџџџџџџџ22
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
l
Kernel_Bias_3/shapeConst*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=
І
"Kernel_Bias_3/RandomStandardNormalRandomStandardNormalKernel_Bias_3/shape*

seed *
T0*
dtype0*&
_output_shapes
:22
*
seed2 

Kernel_Bias_3/mulMul"Kernel_Bias_3/RandomStandardNormalKernel_Bias_3/stddev*&
_output_shapes
:22
*
T0
l
Kernel_Bias_3AddKernel_Bias_3/mulKernel_Bias_3/mean*&
_output_shapes
:22
*
T0


Variable_7
VariableV2*&
_output_shapes
:22
*
	container *
shape:22
*
shared_name *
dtype0
Џ
Variable_7/AssignAssign
Variable_7Kernel_Bias_3*
T0*
_class
loc:@Variable_7*
validate_shape(*&
_output_shapes
:22
*
use_locking(
w
Variable_7/readIdentity
Variable_7*&
_output_shapes
:22
*
T0*
_class
loc:@Variable_7
a
add_3Addconv2d_3Variable_7/read*
T0*/
_output_shapes
:џџџџџџџџџ22

O
relu_3Reluadd_3*/
_output_shapes
:џџџџџџџџџ22
*
T0
g
Kernel_4/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_4/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_4/RandomStandardNormalRandomStandardNormalKernel_4/shape*
dtype0*&
_output_shapes
:



*
seed2 *

seed *
T0
t
Kernel_4/mulMulKernel_4/RandomStandardNormalKernel_4/stddev*&
_output_shapes
:



*
T0
]
Kernel_4AddKernel_4/mulKernel_4/mean*
T0*&
_output_shapes
:






Variable_8
VariableV2*
shared_name *
dtype0*&
_output_shapes
:



*
	container *
shape:




Њ
Variable_8/AssignAssign
Variable_8Kernel_4*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*&
_output_shapes
:




w
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8*&
_output_shapes
:




г
conv2d_4Conv2Drelu_3Variable_8/read*/
_output_shapes
:џџџџџџџџџ22
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
l
Kernel_Bias_4/shapeConst*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Y
Kernel_Bias_4/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_4/RandomStandardNormalRandomStandardNormalKernel_Bias_4/shape*
dtype0*&
_output_shapes
:22
*
seed2 *

seed *
T0

Kernel_Bias_4/mulMul"Kernel_Bias_4/RandomStandardNormalKernel_Bias_4/stddev*
T0*&
_output_shapes
:22

l
Kernel_Bias_4AddKernel_Bias_4/mulKernel_Bias_4/mean*
T0*&
_output_shapes
:22



Variable_9
VariableV2*
dtype0*&
_output_shapes
:22
*
	container *
shape:22
*
shared_name 
Џ
Variable_9/AssignAssign
Variable_9Kernel_Bias_4*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*&
_output_shapes
:22

w
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9*&
_output_shapes
:22

a
add_4Addconv2d_4Variable_9/read*
T0*/
_output_shapes
:џџџџџџџџџ22

O
relu_4Reluadd_4*
T0*/
_output_shapes
:џџџџџџџџџ22

Ѕ
Pool_4MaxPoolrelu_4*/
_output_shapes
:џџџџџџџџџ
*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
g
Kernel_5/shapeConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   
R
Kernel_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_5/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_5/RandomStandardNormalRandomStandardNormalKernel_5/shape*
dtype0*&
_output_shapes
:



*
seed2 *

seed *
T0
t
Kernel_5/mulMulKernel_5/RandomStandardNormalKernel_5/stddev*
T0*&
_output_shapes
:




]
Kernel_5AddKernel_5/mulKernel_5/mean*
T0*&
_output_shapes
:





Variable_10
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
­
Variable_10/AssignAssignVariable_10Kernel_5*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*&
_output_shapes
:




z
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10*&
_output_shapes
:




д
conv2d_5Conv2DPool_4Variable_10/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
	dilations

l
Kernel_Bias_5/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_5/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
Y
Kernel_Bias_5/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_5/RandomStandardNormalRandomStandardNormalKernel_Bias_5/shape*
T0*
dtype0*&
_output_shapes
:
*
seed2 *

seed 

Kernel_Bias_5/mulMul"Kernel_Bias_5/RandomStandardNormalKernel_Bias_5/stddev*&
_output_shapes
:
*
T0
l
Kernel_Bias_5AddKernel_Bias_5/mulKernel_Bias_5/mean*
T0*&
_output_shapes
:


Variable_11
VariableV2*
shape:
*
shared_name *
dtype0*&
_output_shapes
:
*
	container 
В
Variable_11/AssignAssignVariable_11Kernel_Bias_5*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*&
_output_shapes
:

z
Variable_11/readIdentityVariable_11*
T0*
_class
loc:@Variable_11*&
_output_shapes
:

b
add_5Addconv2d_5Variable_11/read*
T0*/
_output_shapes
:џџџџџџџџџ

O
relu_5Reluadd_5*/
_output_shapes
:џџџџџџџџџ
*
T0
g
Kernel_6/shapeConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
R
Kernel_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
Kernel_6/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_6/RandomStandardNormalRandomStandardNormalKernel_6/shape*
T0*
dtype0*&
_output_shapes
:



*
seed2 *

seed 
t
Kernel_6/mulMulKernel_6/RandomStandardNormalKernel_6/stddev*
T0*&
_output_shapes
:




]
Kernel_6AddKernel_6/mulKernel_6/mean*
T0*&
_output_shapes
:





Variable_12
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
­
Variable_12/AssignAssignVariable_12Kernel_6*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(
z
Variable_12/readIdentityVariable_12*
T0*
_class
loc:@Variable_12*&
_output_shapes
:




д
conv2d_6Conv2Drelu_5Variable_12/read*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
l
Kernel_Bias_6/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_6/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_6/RandomStandardNormalRandomStandardNormalKernel_Bias_6/shape*
dtype0*&
_output_shapes
:
*
seed2 *

seed *
T0

Kernel_Bias_6/mulMul"Kernel_Bias_6/RandomStandardNormalKernel_Bias_6/stddev*
T0*&
_output_shapes
:

l
Kernel_Bias_6AddKernel_Bias_6/mulKernel_Bias_6/mean*
T0*&
_output_shapes
:


Variable_13
VariableV2*
shared_name *
dtype0*&
_output_shapes
:
*
	container *
shape:

В
Variable_13/AssignAssignVariable_13Kernel_Bias_6*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_13
z
Variable_13/readIdentityVariable_13*&
_output_shapes
:
*
T0*
_class
loc:@Variable_13
b
add_6Addconv2d_6Variable_13/read*/
_output_shapes
:џџџџџџџџџ
*
T0
O
relu_6Reluadd_6*
T0*/
_output_shapes
:џџџџџџџџџ

g
Kernel_7/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_7/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
T
Kernel_7/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_7/RandomStandardNormalRandomStandardNormalKernel_7/shape*&
_output_shapes
:



*
seed2 *

seed *
T0*
dtype0
t
Kernel_7/mulMulKernel_7/RandomStandardNormalKernel_7/stddev*&
_output_shapes
:



*
T0
]
Kernel_7AddKernel_7/mulKernel_7/mean*&
_output_shapes
:



*
T0

Variable_14
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
­
Variable_14/AssignAssignVariable_14Kernel_7*
_class
loc:@Variable_14*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
z
Variable_14/readIdentityVariable_14*
T0*
_class
loc:@Variable_14*&
_output_shapes
:




д
conv2d_7Conv2Drelu_6Variable_14/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ

l
Kernel_Bias_7/shapeConst*
_output_shapes
:*%
valueB"         
   *
dtype0
W
Kernel_Bias_7/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_7/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_7/RandomStandardNormalRandomStandardNormalKernel_Bias_7/shape*&
_output_shapes
:
*
seed2 *

seed *
T0*
dtype0

Kernel_Bias_7/mulMul"Kernel_Bias_7/RandomStandardNormalKernel_Bias_7/stddev*
T0*&
_output_shapes
:

l
Kernel_Bias_7AddKernel_Bias_7/mulKernel_Bias_7/mean*&
_output_shapes
:
*
T0

Variable_15
VariableV2*
shared_name *
dtype0*&
_output_shapes
:
*
	container *
shape:

В
Variable_15/AssignAssignVariable_15Kernel_Bias_7*
_class
loc:@Variable_15*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0
z
Variable_15/readIdentityVariable_15*
_class
loc:@Variable_15*&
_output_shapes
:
*
T0
b
add_7Addconv2d_7Variable_15/read*
T0*/
_output_shapes
:џџџџџџџџџ

O
relu_7Reluadd_7*
T0*/
_output_shapes
:џџџџџџџџџ

g
Kernel_8/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_8/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
T
Kernel_8/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_8/RandomStandardNormalRandomStandardNormalKernel_8/shape*
T0*
dtype0*&
_output_shapes
:



*
seed2 *

seed 
t
Kernel_8/mulMulKernel_8/RandomStandardNormalKernel_8/stddev*
T0*&
_output_shapes
:




]
Kernel_8AddKernel_8/mulKernel_8/mean*&
_output_shapes
:



*
T0

Variable_16
VariableV2*
shared_name *
dtype0*&
_output_shapes
:



*
	container *
shape:




­
Variable_16/AssignAssignVariable_16Kernel_8*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(
z
Variable_16/readIdentityVariable_16*
T0*
_class
loc:@Variable_16*&
_output_shapes
:




д
conv2d_8Conv2Drelu_7Variable_16/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0
l
Kernel_Bias_8/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_8/stddevConst*
_output_shapes
: *
valueB
 *ЭЬL=*
dtype0
І
"Kernel_Bias_8/RandomStandardNormalRandomStandardNormalKernel_Bias_8/shape*
T0*
dtype0*&
_output_shapes
:
*
seed2 *

seed 

Kernel_Bias_8/mulMul"Kernel_Bias_8/RandomStandardNormalKernel_Bias_8/stddev*
T0*&
_output_shapes
:

l
Kernel_Bias_8AddKernel_Bias_8/mulKernel_Bias_8/mean*&
_output_shapes
:
*
T0

Variable_17
VariableV2*
shared_name *
dtype0*&
_output_shapes
:
*
	container *
shape:

В
Variable_17/AssignAssignVariable_17Kernel_Bias_8*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*&
_output_shapes
:

z
Variable_17/readIdentityVariable_17*
T0*
_class
loc:@Variable_17*&
_output_shapes
:

b
add_8Addconv2d_8Variable_17/read*
T0*/
_output_shapes
:џџџџџџџџџ

O
relu_8Reluadd_8*
T0*/
_output_shapes
:џџџџџџџџџ

Ѕ
Pool_8MaxPoolrelu_8*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
T0
g
Kernel_9/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
R
Kernel_9/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
T
Kernel_9/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_9/RandomStandardNormalRandomStandardNormalKernel_9/shape*

seed *
T0*
dtype0*&
_output_shapes
:



*
seed2 
t
Kernel_9/mulMulKernel_9/RandomStandardNormalKernel_9/stddev*
T0*&
_output_shapes
:




]
Kernel_9AddKernel_9/mulKernel_9/mean*
T0*&
_output_shapes
:





Variable_18
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
­
Variable_18/AssignAssignVariable_18Kernel_9*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_18
z
Variable_18/readIdentityVariable_18*
T0*
_class
loc:@Variable_18*&
_output_shapes
:




д
conv2d_9Conv2DPool_8Variable_18/read*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
l
Kernel_Bias_9/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
W
Kernel_Bias_9/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
Kernel_Bias_9/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
І
"Kernel_Bias_9/RandomStandardNormalRandomStandardNormalKernel_Bias_9/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 

Kernel_Bias_9/mulMul"Kernel_Bias_9/RandomStandardNormalKernel_Bias_9/stddev*
T0*&
_output_shapes
:

l
Kernel_Bias_9AddKernel_Bias_9/mulKernel_Bias_9/mean*&
_output_shapes
:
*
T0

Variable_19
VariableV2*
shared_name *
dtype0*&
_output_shapes
:
*
	container *
shape:

В
Variable_19/AssignAssignVariable_19Kernel_Bias_9*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_19
z
Variable_19/readIdentityVariable_19*
T0*
_class
loc:@Variable_19*&
_output_shapes
:

b
add_9Addconv2d_9Variable_19/read*
T0*/
_output_shapes
:џџџџџџџџџ

O
relu_9Reluadd_9*
T0*/
_output_shapes
:џџџџџџџџџ

h
Kernel_10/shapeConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
S
Kernel_10/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
U
Kernel_10/stddevConst*
_output_shapes
: *
valueB
 *ЭЬL=*
dtype0

Kernel_10/RandomStandardNormalRandomStandardNormalKernel_10/shape*

seed *
T0*
dtype0*&
_output_shapes
:



*
seed2 
w
Kernel_10/mulMulKernel_10/RandomStandardNormalKernel_10/stddev*
T0*&
_output_shapes
:




`
	Kernel_10AddKernel_10/mulKernel_10/mean*
T0*&
_output_shapes
:





Variable_20
VariableV2*
shape:



*
shared_name *
dtype0*&
_output_shapes
:



*
	container 
Ў
Variable_20/AssignAssignVariable_20	Kernel_10*
_class
loc:@Variable_20*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
z
Variable_20/readIdentityVariable_20*
_class
loc:@Variable_20*&
_output_shapes
:



*
T0
е
	conv2d_10Conv2Drelu_9Variable_20/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0
m
Kernel_Bias_10/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
X
Kernel_Bias_10/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
Kernel_Bias_10/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
Ј
#Kernel_Bias_10/RandomStandardNormalRandomStandardNormalKernel_Bias_10/shape*
T0*
dtype0*&
_output_shapes
:
*
seed2 *

seed 

Kernel_Bias_10/mulMul#Kernel_Bias_10/RandomStandardNormalKernel_Bias_10/stddev*&
_output_shapes
:
*
T0
o
Kernel_Bias_10AddKernel_Bias_10/mulKernel_Bias_10/mean*
T0*&
_output_shapes
:


Variable_21
VariableV2*
shape:
*
shared_name *
dtype0*&
_output_shapes
:
*
	container 
Г
Variable_21/AssignAssignVariable_21Kernel_Bias_10*
use_locking(*
T0*
_class
loc:@Variable_21*
validate_shape(*&
_output_shapes
:

z
Variable_21/readIdentityVariable_21*
_class
loc:@Variable_21*&
_output_shapes
:
*
T0
d
add_10Add	conv2d_10Variable_21/read*
T0*/
_output_shapes
:џџџџџџџџџ

Q
relu_10Reluadd_10*/
_output_shapes
:џџџџџџџџџ
*
T0
h
Kernel_11/shapeConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
S
Kernel_11/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
U
Kernel_11/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

Kernel_11/RandomStandardNormalRandomStandardNormalKernel_11/shape*
T0*
dtype0*&
_output_shapes
:



*
seed2 *

seed 
w
Kernel_11/mulMulKernel_11/RandomStandardNormalKernel_11/stddev*
T0*&
_output_shapes
:




`
	Kernel_11AddKernel_11/mulKernel_11/mean*&
_output_shapes
:



*
T0

Variable_22
VariableV2*
shape:



*
shared_name *
dtype0*&
_output_shapes
:



*
	container 
Ў
Variable_22/AssignAssignVariable_22	Kernel_11*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
:



*
use_locking(
z
Variable_22/readIdentityVariable_22*
T0*
_class
loc:@Variable_22*&
_output_shapes
:




ж
	conv2d_11Conv2Drelu_10Variable_22/read*/
_output_shapes
:џџџџџџџџџ
*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
m
Kernel_Bias_11/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
X
Kernel_Bias_11/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
Kernel_Bias_11/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
Ј
#Kernel_Bias_11/RandomStandardNormalRandomStandardNormalKernel_Bias_11/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 

Kernel_Bias_11/mulMul#Kernel_Bias_11/RandomStandardNormalKernel_Bias_11/stddev*
T0*&
_output_shapes
:

o
Kernel_Bias_11AddKernel_Bias_11/mulKernel_Bias_11/mean*
T0*&
_output_shapes
:


Variable_23
VariableV2*
shared_name *
dtype0*&
_output_shapes
:
*
	container *
shape:

Г
Variable_23/AssignAssignVariable_23Kernel_Bias_11*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_23
z
Variable_23/readIdentityVariable_23*
T0*
_class
loc:@Variable_23*&
_output_shapes
:

d
add_11Add	conv2d_11Variable_23/read*
T0*/
_output_shapes
:џџџџџџџџџ

Q
relu_11Reluadd_11*
T0*/
_output_shapes
:џџџџџџџџџ

h
Kernel_12/shapeConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   
S
Kernel_12/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
U
Kernel_12/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL=

Kernel_12/RandomStandardNormalRandomStandardNormalKernel_12/shape*
dtype0*&
_output_shapes
:



*
seed2 *

seed *
T0
w
Kernel_12/mulMulKernel_12/RandomStandardNormalKernel_12/stddev*
T0*&
_output_shapes
:




`
	Kernel_12AddKernel_12/mulKernel_12/mean*&
_output_shapes
:



*
T0

Variable_24
VariableV2*
dtype0*&
_output_shapes
:



*
	container *
shape:



*
shared_name 
Ў
Variable_24/AssignAssignVariable_24	Kernel_12*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(*&
_output_shapes
:




z
Variable_24/readIdentityVariable_24*
T0*
_class
loc:@Variable_24*&
_output_shapes
:




ж
	conv2d_12Conv2Drelu_11Variable_24/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ

m
Kernel_Bias_12/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
X
Kernel_Bias_12/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Z
Kernel_Bias_12/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 
Ј
#Kernel_Bias_12/RandomStandardNormalRandomStandardNormalKernel_Bias_12/shape*
dtype0*&
_output_shapes
:
*
seed2 *

seed *
T0

Kernel_Bias_12/mulMul#Kernel_Bias_12/RandomStandardNormalKernel_Bias_12/stddev*
T0*&
_output_shapes
:

o
Kernel_Bias_12AddKernel_Bias_12/mulKernel_Bias_12/mean*&
_output_shapes
:
*
T0

Variable_25
VariableV2*
dtype0*&
_output_shapes
:
*
	container *
shape:
*
shared_name 
Г
Variable_25/AssignAssignVariable_25Kernel_Bias_12*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(*&
_output_shapes
:

z
Variable_25/readIdentityVariable_25*
T0*
_class
loc:@Variable_25*&
_output_shapes
:

d
add_12Add	conv2d_12Variable_25/read*/
_output_shapes
:џџџџџџџџџ
*
T0
Q
relu_12Reluadd_12*
T0*/
_output_shapes
:џџџџџџџџџ

Ї
Pool_12MaxPoolrelu_12*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ
*
T0
^
Reshape/shapeConst*
valueB"   џџџџ*
dtype0*
_output_shapes
:
j
ReshapeReshapePool_12Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
`
FC_Weight/shapeConst*
valueB"ъ  	   *
dtype0*
_output_shapes
:
S
FC_Weight/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
U
FC_Weight/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

FC_Weight/RandomStandardNormalRandomStandardNormalFC_Weight/shape*
T0*
dtype0*
_output_shapes
:	ъ	*
seed2 *

seed 
p
FC_Weight/mulMulFC_Weight/RandomStandardNormalFC_Weight/stddev*
T0*
_output_shapes
:	ъ	
Y
	FC_WeightAddFC_Weight/mulFC_Weight/mean*
T0*
_output_shapes
:	ъ	

Variable_26
VariableV2*
dtype0*
_output_shapes
:	ъ	*
	container *
shape:	ъ	*
shared_name 
Ї
Variable_26/AssignAssignVariable_26	FC_Weight*
use_locking(*
T0*
_class
loc:@Variable_26*
validate_shape(*
_output_shapes
:	ъ	
s
Variable_26/readIdentityVariable_26*
_class
loc:@Variable_26*
_output_shapes
:	ъ	*
T0
^
FC_Bias/shapeConst*
valueB"   	   *
dtype0*
_output_shapes
:
Q
FC_Bias/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
S
FC_Bias/stddevConst*
valueB
 *ЭЬL=*
dtype0*
_output_shapes
: 

FC_Bias/RandomStandardNormalRandomStandardNormalFC_Bias/shape*
dtype0*
_output_shapes

:	*
seed2 *

seed *
T0
i
FC_Bias/mulMulFC_Bias/RandomStandardNormalFC_Bias/stddev*
T0*
_output_shapes

:	
R
FC_BiasAddFC_Bias/mulFC_Bias/mean*
T0*
_output_shapes

:	

Variable_27
VariableV2*
shape
:	*
shared_name *
dtype0*
_output_shapes

:	*
	container 
Є
Variable_27/AssignAssignVariable_27FC_Bias*
_class
loc:@Variable_27*
validate_shape(*
_output_shapes

:	*
use_locking(*
T0
r
Variable_27/readIdentityVariable_27*
T0*
_class
loc:@Variable_27*
_output_shapes

:	
}
	FC_MatMulMatMulReshapeVariable_26/read*
_output_shapes

:	*
transpose_a( *
transpose_b( *
T0
S
add_13Add	FC_MatMulVariable_27/read*
T0*
_output_shapes

:	
V
dropout/keep_probConst*
valueB
 *33s?*
dtype0*
_output_shapes
: 
^
dropout/ShapeConst*
valueB"   	   *
dtype0*
_output_shapes
:
_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *
T0*
dtype0*
_output_shapes

:	*
seed2 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*
_output_shapes

:	
~
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
_output_shapes

:	*
T0
f
dropout/addAdddropout/keep_probdropout/random_uniform*
_output_shapes

:	*
T0
L
dropout/FloorFloordropout/add*
_output_shapes

:	*
T0
Z
dropout/divRealDivadd_13dropout/keep_prob*
T0*
_output_shapes

:	
W
dropout/mulMuldropout/divdropout/Floor*
T0*
_output_shapes

:	
Y
Label_Maker/on_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Z
Label_Maker/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
S
Label_Maker/depthConst*
value	B :	*
dtype0*
_output_shapes
: 
І
Label_MakerOneHotPlaceholder_yLabel_Maker/depthLabel_Maker/on_valueLabel_Maker/off_value*
T0*
TI0*
axisџџџџџџџџџ*
_output_shapes
:
S
Loss_SOFTMAX/RankConst*
_output_shapes
: *
value	B :*
dtype0
c
Loss_SOFTMAX/ShapeConst*
valueB"   	   *
dtype0*
_output_shapes
:
U
Loss_SOFTMAX/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
e
Loss_SOFTMAX/Shape_1Const*
valueB"   	   *
dtype0*
_output_shapes
:
T
Loss_SOFTMAX/Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
a
Loss_SOFTMAX/SubSubLoss_SOFTMAX/Rank_1Loss_SOFTMAX/Sub/y*
T0*
_output_shapes
: 
l
Loss_SOFTMAX/Slice/beginPackLoss_SOFTMAX/Sub*
T0*

axis *
N*
_output_shapes
:
a
Loss_SOFTMAX/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0

Loss_SOFTMAX/SliceSliceLoss_SOFTMAX/Shape_1Loss_SOFTMAX/Slice/beginLoss_SOFTMAX/Slice/size*
Index0*
T0*
_output_shapes
:
o
Loss_SOFTMAX/concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Z
Loss_SOFTMAX/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
Ѕ
Loss_SOFTMAX/concatConcatV2Loss_SOFTMAX/concat/values_0Loss_SOFTMAX/SliceLoss_SOFTMAX/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
s
Loss_SOFTMAX/ReshapeReshapeadd_13Loss_SOFTMAX/concat*
_output_shapes

:	*
T0*
Tshape0
I
Loss_SOFTMAX/Rank_2RankLabel_Maker*
_output_shapes
: *
T0
h
Loss_SOFTMAX/Shape_2ShapeLabel_Maker*
T0*
out_type0*#
_output_shapes
:џџџџџџџџџ
V
Loss_SOFTMAX/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
e
Loss_SOFTMAX/Sub_1SubLoss_SOFTMAX/Rank_2Loss_SOFTMAX/Sub_1/y*
T0*
_output_shapes
: 
p
Loss_SOFTMAX/Slice_1/beginPackLoss_SOFTMAX/Sub_1*
_output_shapes
:*
T0*

axis *
N
c
Loss_SOFTMAX/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

Loss_SOFTMAX/Slice_1SliceLoss_SOFTMAX/Shape_2Loss_SOFTMAX/Slice_1/beginLoss_SOFTMAX/Slice_1/size*
Index0*
T0*
_output_shapes
:
q
Loss_SOFTMAX/concat_1/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
\
Loss_SOFTMAX/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
­
Loss_SOFTMAX/concat_1ConcatV2Loss_SOFTMAX/concat_1/values_0Loss_SOFTMAX/Slice_1Loss_SOFTMAX/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N

Loss_SOFTMAX/Reshape_1ReshapeLabel_MakerLoss_SOFTMAX/concat_1*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Loss_SOFTMAXSoftmaxCrossEntropyWithLogitsLoss_SOFTMAX/ReshapeLoss_SOFTMAX/Reshape_1*
T0*$
_output_shapes
::	
V
Loss_SOFTMAX/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
c
Loss_SOFTMAX/Sub_2SubLoss_SOFTMAX/RankLoss_SOFTMAX/Sub_2/y*
_output_shapes
: *
T0
d
Loss_SOFTMAX/Slice_2/beginConst*
_output_shapes
:*
valueB: *
dtype0
o
Loss_SOFTMAX/Slice_2/sizePackLoss_SOFTMAX/Sub_2*

axis *
N*
_output_shapes
:*
T0
Ѓ
Loss_SOFTMAX/Slice_2SliceLoss_SOFTMAX/ShapeLoss_SOFTMAX/Slice_2/beginLoss_SOFTMAX/Slice_2/size*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ
x
Loss_SOFTMAX/Reshape_2ReshapeLoss_SOFTMAXLoss_SOFTMAX/Slice_2*
_output_shapes
:*
T0*
Tshape0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
p
Reduce_MeanMeanLoss_SOFTMAX/Reshape_2Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
r
(gradients/Reduce_Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0

"gradients/Reduce_Mean_grad/ReshapeReshapegradients/Fill(gradients/Reduce_Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
j
 gradients/Reduce_Mean_grad/ConstConst*
_output_shapes
:*
valueB:*
dtype0
Є
gradients/Reduce_Mean_grad/TileTile"gradients/Reduce_Mean_grad/Reshape gradients/Reduce_Mean_grad/Const*
_output_shapes
:*

Tmultiples0*
T0
g
"gradients/Reduce_Mean_grad/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"gradients/Reduce_Mean_grad/truedivRealDivgradients/Reduce_Mean_grad/Tile"gradients/Reduce_Mean_grad/Const_1*
T0*
_output_shapes
:
u
+gradients/Loss_SOFTMAX/Reshape_2_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0
М
-gradients/Loss_SOFTMAX/Reshape_2_grad/ReshapeReshape"gradients/Reduce_Mean_grad/truediv+gradients/Loss_SOFTMAX/Reshape_2_grad/Shape*
T0*
Tshape0*
_output_shapes
:
Z
gradients/zeros_like	ZerosLikeLoss_SOFTMAX:1*
T0*
_output_shapes

:	
u
*gradients/Loss_SOFTMAX_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ф
&gradients/Loss_SOFTMAX_grad/ExpandDims
ExpandDims-gradients/Loss_SOFTMAX/Reshape_2_grad/Reshape*gradients/Loss_SOFTMAX_grad/ExpandDims/dim*
_output_shapes

:*

Tdim0*
T0

gradients/Loss_SOFTMAX_grad/mulMul&gradients/Loss_SOFTMAX_grad/ExpandDimsLoss_SOFTMAX:1*
T0*
_output_shapes

:	
s
&gradients/Loss_SOFTMAX_grad/LogSoftmax
LogSoftmaxLoss_SOFTMAX/Reshape*
T0*
_output_shapes

:	
w
gradients/Loss_SOFTMAX_grad/NegNeg&gradients/Loss_SOFTMAX_grad/LogSoftmax*
_output_shapes

:	*
T0
w
,gradients/Loss_SOFTMAX_grad/ExpandDims_1/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ш
(gradients/Loss_SOFTMAX_grad/ExpandDims_1
ExpandDims-gradients/Loss_SOFTMAX/Reshape_2_grad/Reshape,gradients/Loss_SOFTMAX_grad/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:

!gradients/Loss_SOFTMAX_grad/mul_1Mul(gradients/Loss_SOFTMAX_grad/ExpandDims_1gradients/Loss_SOFTMAX_grad/Neg*
T0*
_output_shapes

:	
z
,gradients/Loss_SOFTMAX_grad/tuple/group_depsNoOp ^gradients/Loss_SOFTMAX_grad/mul"^gradients/Loss_SOFTMAX_grad/mul_1
э
4gradients/Loss_SOFTMAX_grad/tuple/control_dependencyIdentitygradients/Loss_SOFTMAX_grad/mul-^gradients/Loss_SOFTMAX_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/Loss_SOFTMAX_grad/mul*
_output_shapes

:	
ѓ
6gradients/Loss_SOFTMAX_grad/tuple/control_dependency_1Identity!gradients/Loss_SOFTMAX_grad/mul_1-^gradients/Loss_SOFTMAX_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/Loss_SOFTMAX_grad/mul_1*
_output_shapes

:	
z
)gradients/Loss_SOFTMAX/Reshape_grad/ShapeConst*
valueB"   	   *
dtype0*
_output_shapes
:
Ю
+gradients/Loss_SOFTMAX/Reshape_grad/ReshapeReshape4gradients/Loss_SOFTMAX_grad/tuple/control_dependency)gradients/Loss_SOFTMAX/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes

:	
\
&gradients/add_13_grad/tuple/group_depsNoOp,^gradients/Loss_SOFTMAX/Reshape_grad/Reshape
љ
.gradients/add_13_grad/tuple/control_dependencyIdentity+gradients/Loss_SOFTMAX/Reshape_grad/Reshape'^gradients/add_13_grad/tuple/group_deps*
_output_shapes

:	*
T0*>
_class4
20loc:@gradients/Loss_SOFTMAX/Reshape_grad/Reshape
ћ
0gradients/add_13_grad/tuple/control_dependency_1Identity+gradients/Loss_SOFTMAX/Reshape_grad/Reshape'^gradients/add_13_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Loss_SOFTMAX/Reshape_grad/Reshape*
_output_shapes

:	
Л
gradients/FC_MatMul_grad/MatMulMatMul.gradients/add_13_grad/tuple/control_dependencyVariable_26/read*
_output_shapes
:	ъ*
transpose_a( *
transpose_b(*
T0
М
!gradients/FC_MatMul_grad/MatMul_1MatMulReshape.gradients/add_13_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ	*
transpose_a(*
transpose_b( *
T0
w
)gradients/FC_MatMul_grad/tuple/group_depsNoOp ^gradients/FC_MatMul_grad/MatMul"^gradients/FC_MatMul_grad/MatMul_1
ш
1gradients/FC_MatMul_grad/tuple/control_dependencyIdentitygradients/FC_MatMul_grad/MatMul*^gradients/FC_MatMul_grad/tuple/group_deps*
_output_shapes
:	ъ*
T0*2
_class(
&$loc:@gradients/FC_MatMul_grad/MatMul
ю
3gradients/FC_MatMul_grad/tuple/control_dependency_1Identity!gradients/FC_MatMul_grad/MatMul_1*^gradients/FC_MatMul_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/FC_MatMul_grad/MatMul_1*
_output_shapes
:	ъ	
c
gradients/Reshape_grad/ShapeShapePool_12*
out_type0*
_output_shapes
:*
T0
Й
gradients/Reshape_grad/ReshapeReshape1gradients/FC_MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*&
_output_shapes
:
*
T0*
Tshape0
я
"gradients/Pool_12_grad/MaxPoolGradMaxPoolGradrelu_12Pool_12gradients/Reshape_grad/Reshape*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ


gradients/relu_12_grad/ReluGradReluGrad"gradients/Pool_12_grad/MaxPoolGradrelu_12*
T0*/
_output_shapes
:џџџџџџџџџ

d
gradients/add_12_grad/ShapeShape	conv2d_12*
_output_shapes
:*
T0*
out_type0
v
gradients/add_12_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
Н
+gradients/add_12_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_12_grad/Shapegradients/add_12_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ў
gradients/add_12_grad/SumSumgradients/relu_12_grad/ReluGrad+gradients/add_12_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ј
gradients/add_12_grad/ReshapeReshapegradients/add_12_grad/Sumgradients/add_12_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ

В
gradients/add_12_grad/Sum_1Sumgradients/relu_12_grad/ReluGrad-gradients/add_12_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_12_grad/Reshape_1Reshapegradients/add_12_grad/Sum_1gradients/add_12_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

p
&gradients/add_12_grad/tuple/group_depsNoOp^gradients/add_12_grad/Reshape ^gradients/add_12_grad/Reshape_1
ю
.gradients/add_12_grad/tuple/control_dependencyIdentitygradients/add_12_grad/Reshape'^gradients/add_12_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*0
_class&
$"loc:@gradients/add_12_grad/Reshape
ы
0gradients/add_12_grad/tuple/control_dependency_1Identitygradients/add_12_grad/Reshape_1'^gradients/add_12_grad/tuple/group_deps*&
_output_shapes
:
*
T0*2
_class(
&$loc:@gradients/add_12_grad/Reshape_1

gradients/conv2d_12_grad/ShapeNShapeNrelu_11Variable_24/read*
T0*
out_type0*
N* 
_output_shapes
::
w
gradients/conv2d_12_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
щ
,gradients/conv2d_12_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_12_grad/ShapeNVariable_24/read.gradients/add_12_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Н
-gradients/conv2d_12_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_11gradients/conv2d_12_grad/Const.gradients/add_12_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:





)gradients/conv2d_12_grad/tuple/group_depsNoOp-^gradients/conv2d_12_grad/Conv2DBackpropInput.^gradients/conv2d_12_grad/Conv2DBackpropFilter

1gradients/conv2d_12_grad/tuple/control_dependencyIdentity,gradients/conv2d_12_grad/Conv2DBackpropInput*^gradients/conv2d_12_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_12_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


3gradients/conv2d_12_grad/tuple/control_dependency_1Identity-gradients/conv2d_12_grad/Conv2DBackpropFilter*^gradients/conv2d_12_grad/tuple/group_deps*&
_output_shapes
:



*
T0*@
_class6
42loc:@gradients/conv2d_12_grad/Conv2DBackpropFilter
Ё
gradients/relu_11_grad/ReluGradReluGrad1gradients/conv2d_12_grad/tuple/control_dependencyrelu_11*
T0*/
_output_shapes
:џџџџџџџџџ

d
gradients/add_11_grad/ShapeShape	conv2d_11*
T0*
out_type0*
_output_shapes
:
v
gradients/add_11_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
Н
+gradients/add_11_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_11_grad/Shapegradients/add_11_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ў
gradients/add_11_grad/SumSumgradients/relu_11_grad/ReluGrad+gradients/add_11_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ј
gradients/add_11_grad/ReshapeReshapegradients/add_11_grad/Sumgradients/add_11_grad/Shape*/
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0
В
gradients/add_11_grad/Sum_1Sumgradients/relu_11_grad/ReluGrad-gradients/add_11_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_11_grad/Reshape_1Reshapegradients/add_11_grad/Sum_1gradients/add_11_grad/Shape_1*
Tshape0*&
_output_shapes
:
*
T0
p
&gradients/add_11_grad/tuple/group_depsNoOp^gradients/add_11_grad/Reshape ^gradients/add_11_grad/Reshape_1
ю
.gradients/add_11_grad/tuple/control_dependencyIdentitygradients/add_11_grad/Reshape'^gradients/add_11_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_11_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ы
0gradients/add_11_grad/tuple/control_dependency_1Identitygradients/add_11_grad/Reshape_1'^gradients/add_11_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_11_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_11_grad/ShapeNShapeNrelu_10Variable_22/read*
out_type0*
N* 
_output_shapes
::*
T0
w
gradients/conv2d_11_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
щ
,gradients/conv2d_11_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_11_grad/ShapeNVariable_22/read.gradients/add_11_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Н
-gradients/conv2d_11_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_10gradients/conv2d_11_grad/Const.gradients/add_11_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

)gradients/conv2d_11_grad/tuple/group_depsNoOp-^gradients/conv2d_11_grad/Conv2DBackpropInput.^gradients/conv2d_11_grad/Conv2DBackpropFilter

1gradients/conv2d_11_grad/tuple/control_dependencyIdentity,gradients/conv2d_11_grad/Conv2DBackpropInput*^gradients/conv2d_11_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_11_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


3gradients/conv2d_11_grad/tuple/control_dependency_1Identity-gradients/conv2d_11_grad/Conv2DBackpropFilter*^gradients/conv2d_11_grad/tuple/group_deps*&
_output_shapes
:



*
T0*@
_class6
42loc:@gradients/conv2d_11_grad/Conv2DBackpropFilter
Ё
gradients/relu_10_grad/ReluGradReluGrad1gradients/conv2d_11_grad/tuple/control_dependencyrelu_10*
T0*/
_output_shapes
:џџџџџџџџџ

d
gradients/add_10_grad/ShapeShape	conv2d_10*
_output_shapes
:*
T0*
out_type0
v
gradients/add_10_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
Н
+gradients/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_10_grad/Shapegradients/add_10_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
gradients/add_10_grad/SumSumgradients/relu_10_grad/ReluGrad+gradients/add_10_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ј
gradients/add_10_grad/ReshapeReshapegradients/add_10_grad/Sumgradients/add_10_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ
*
T0
В
gradients/add_10_grad/Sum_1Sumgradients/relu_10_grad/ReluGrad-gradients/add_10_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_10_grad/Reshape_1Reshapegradients/add_10_grad/Sum_1gradients/add_10_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

p
&gradients/add_10_grad/tuple/group_depsNoOp^gradients/add_10_grad/Reshape ^gradients/add_10_grad/Reshape_1
ю
.gradients/add_10_grad/tuple/control_dependencyIdentitygradients/add_10_grad/Reshape'^gradients/add_10_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_10_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ы
0gradients/add_10_grad/tuple/control_dependency_1Identitygradients/add_10_grad/Reshape_1'^gradients/add_10_grad/tuple/group_deps*&
_output_shapes
:
*
T0*2
_class(
&$loc:@gradients/add_10_grad/Reshape_1

gradients/conv2d_10_grad/ShapeNShapeNrelu_9Variable_20/read*
T0*
out_type0*
N* 
_output_shapes
::
w
gradients/conv2d_10_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
щ
,gradients/conv2d_10_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_10_grad/ShapeNVariable_20/read.gradients/add_10_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
М
-gradients/conv2d_10_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_9gradients/conv2d_10_grad/Const.gradients/add_10_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:





)gradients/conv2d_10_grad/tuple/group_depsNoOp-^gradients/conv2d_10_grad/Conv2DBackpropInput.^gradients/conv2d_10_grad/Conv2DBackpropFilter

1gradients/conv2d_10_grad/tuple/control_dependencyIdentity,gradients/conv2d_10_grad/Conv2DBackpropInput*^gradients/conv2d_10_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_10_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


3gradients/conv2d_10_grad/tuple/control_dependency_1Identity-gradients/conv2d_10_grad/Conv2DBackpropFilter*^gradients/conv2d_10_grad/tuple/group_deps*&
_output_shapes
:



*
T0*@
_class6
42loc:@gradients/conv2d_10_grad/Conv2DBackpropFilter

gradients/relu_9_grad/ReluGradReluGrad1gradients/conv2d_10_grad/tuple/control_dependencyrelu_9*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_9_grad/ShapeShapeconv2d_9*
_output_shapes
:*
T0*
out_type0
u
gradients/add_9_grad/Shape_1Const*
dtype0*
_output_shapes
:*%
valueB"         
   
К
*gradients/add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_9_grad/Shapegradients/add_9_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_9_grad/SumSumgradients/relu_9_grad/ReluGrad*gradients/add_9_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_9_grad/ReshapeReshapegradients/add_9_grad/Sumgradients/add_9_grad/Shape*/
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0
Џ
gradients/add_9_grad/Sum_1Sumgradients/relu_9_grad/ReluGrad,gradients/add_9_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ђ
gradients/add_9_grad/Reshape_1Reshapegradients/add_9_grad/Sum_1gradients/add_9_grad/Shape_1*&
_output_shapes
:
*
T0*
Tshape0
m
%gradients/add_9_grad/tuple/group_depsNoOp^gradients/add_9_grad/Reshape^gradients/add_9_grad/Reshape_1
ъ
-gradients/add_9_grad/tuple/control_dependencyIdentitygradients/add_9_grad/Reshape&^gradients/add_9_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_9_grad/Reshape*/
_output_shapes
:џџџџџџџџџ
*
T0
ч
/gradients/add_9_grad/tuple/control_dependency_1Identitygradients/add_9_grad/Reshape_1&^gradients/add_9_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_9_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_9_grad/ShapeNShapeNPool_8Variable_18/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_9_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
ц
+gradients/conv2d_9_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_9_grad/ShapeNVariable_18/read-gradients/add_9_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC
Й
,gradients/conv2d_9_grad/Conv2DBackpropFilterConv2DBackpropFilterPool_8gradients/conv2d_9_grad/Const-gradients/add_9_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

(gradients/conv2d_9_grad/tuple/group_depsNoOp,^gradients/conv2d_9_grad/Conv2DBackpropInput-^gradients/conv2d_9_grad/Conv2DBackpropFilter

0gradients/conv2d_9_grad/tuple/control_dependencyIdentity+gradients/conv2d_9_grad/Conv2DBackpropInput)^gradients/conv2d_9_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_9_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


2gradients/conv2d_9_grad/tuple/control_dependency_1Identity,gradients/conv2d_9_grad/Conv2DBackpropFilter)^gradients/conv2d_9_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_9_grad/Conv2DBackpropFilter*&
_output_shapes
:




ў
!gradients/Pool_8_grad/MaxPoolGradMaxPoolGradrelu_8Pool_80gradients/conv2d_9_grad/tuple/control_dependency*/
_output_shapes
:џџџџџџџџџ
*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME

gradients/relu_8_grad/ReluGradReluGrad!gradients/Pool_8_grad/MaxPoolGradrelu_8*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_8_grad/ShapeShapeconv2d_8*
_output_shapes
:*
T0*
out_type0
u
gradients/add_8_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
К
*gradients/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_8_grad/Shapegradients/add_8_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_8_grad/SumSumgradients/relu_8_grad/ReluGrad*gradients/add_8_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_8_grad/ReshapeReshapegradients/add_8_grad/Sumgradients/add_8_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ

Џ
gradients/add_8_grad/Sum_1Sumgradients/relu_8_grad/ReluGrad,gradients/add_8_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ђ
gradients/add_8_grad/Reshape_1Reshapegradients/add_8_grad/Sum_1gradients/add_8_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

m
%gradients/add_8_grad/tuple/group_depsNoOp^gradients/add_8_grad/Reshape^gradients/add_8_grad/Reshape_1
ъ
-gradients/add_8_grad/tuple/control_dependencyIdentitygradients/add_8_grad/Reshape&^gradients/add_8_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_8_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ч
/gradients/add_8_grad/tuple/control_dependency_1Identitygradients/add_8_grad/Reshape_1&^gradients/add_8_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_8_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_8_grad/ShapeNShapeNrelu_7Variable_16/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_8_grad/ConstConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
ц
+gradients/conv2d_8_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_8_grad/ShapeNVariable_16/read-gradients/add_8_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC
Й
,gradients/conv2d_8_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_7gradients/conv2d_8_grad/Const-gradients/add_8_grad/tuple/control_dependency*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/conv2d_8_grad/tuple/group_depsNoOp,^gradients/conv2d_8_grad/Conv2DBackpropInput-^gradients/conv2d_8_grad/Conv2DBackpropFilter

0gradients/conv2d_8_grad/tuple/control_dependencyIdentity+gradients/conv2d_8_grad/Conv2DBackpropInput)^gradients/conv2d_8_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*>
_class4
20loc:@gradients/conv2d_8_grad/Conv2DBackpropInput

2gradients/conv2d_8_grad/tuple/control_dependency_1Identity,gradients/conv2d_8_grad/Conv2DBackpropFilter)^gradients/conv2d_8_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_8_grad/Conv2DBackpropFilter*&
_output_shapes
:





gradients/relu_7_grad/ReluGradReluGrad0gradients/conv2d_8_grad/tuple/control_dependencyrelu_7*/
_output_shapes
:џџџџџџџџџ
*
T0
b
gradients/add_7_grad/ShapeShapeconv2d_7*
T0*
out_type0*
_output_shapes
:
u
gradients/add_7_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
К
*gradients/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_7_grad/Shapegradients/add_7_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_7_grad/SumSumgradients/relu_7_grad/ReluGrad*gradients/add_7_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ѕ
gradients/add_7_grad/ReshapeReshapegradients/add_7_grad/Sumgradients/add_7_grad/Shape*/
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0
Џ
gradients/add_7_grad/Sum_1Sumgradients/relu_7_grad/ReluGrad,gradients/add_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ђ
gradients/add_7_grad/Reshape_1Reshapegradients/add_7_grad/Sum_1gradients/add_7_grad/Shape_1*&
_output_shapes
:
*
T0*
Tshape0
m
%gradients/add_7_grad/tuple/group_depsNoOp^gradients/add_7_grad/Reshape^gradients/add_7_grad/Reshape_1
ъ
-gradients/add_7_grad/tuple/control_dependencyIdentitygradients/add_7_grad/Reshape&^gradients/add_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_7_grad/Reshape*/
_output_shapes
:џџџџџџџџџ

ч
/gradients/add_7_grad/tuple/control_dependency_1Identitygradients/add_7_grad/Reshape_1&^gradients/add_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_7_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_7_grad/ShapeNShapeNrelu_6Variable_14/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_7_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   
ц
+gradients/conv2d_7_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_7_grad/ShapeNVariable_14/read-gradients/add_7_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Й
,gradients/conv2d_7_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_6gradients/conv2d_7_grad/Const-gradients/add_7_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

(gradients/conv2d_7_grad/tuple/group_depsNoOp,^gradients/conv2d_7_grad/Conv2DBackpropInput-^gradients/conv2d_7_grad/Conv2DBackpropFilter

0gradients/conv2d_7_grad/tuple/control_dependencyIdentity+gradients/conv2d_7_grad/Conv2DBackpropInput)^gradients/conv2d_7_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_7_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


2gradients/conv2d_7_grad/tuple/control_dependency_1Identity,gradients/conv2d_7_grad/Conv2DBackpropFilter)^gradients/conv2d_7_grad/tuple/group_deps*&
_output_shapes
:



*
T0*?
_class5
31loc:@gradients/conv2d_7_grad/Conv2DBackpropFilter

gradients/relu_6_grad/ReluGradReluGrad0gradients/conv2d_7_grad/tuple/control_dependencyrelu_6*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_6_grad/ShapeShapeconv2d_6*
_output_shapes
:*
T0*
out_type0
u
gradients/add_6_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
К
*gradients/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_6_grad/Shapegradients/add_6_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_6_grad/SumSumgradients/relu_6_grad/ReluGrad*gradients/add_6_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_6_grad/ReshapeReshapegradients/add_6_grad/Sumgradients/add_6_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ
*
T0
Џ
gradients/add_6_grad/Sum_1Sumgradients/relu_6_grad/ReluGrad,gradients/add_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ђ
gradients/add_6_grad/Reshape_1Reshapegradients/add_6_grad/Sum_1gradients/add_6_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

m
%gradients/add_6_grad/tuple/group_depsNoOp^gradients/add_6_grad/Reshape^gradients/add_6_grad/Reshape_1
ъ
-gradients/add_6_grad/tuple/control_dependencyIdentitygradients/add_6_grad/Reshape&^gradients/add_6_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_6_grad/Reshape*/
_output_shapes
:џџџџџџџџџ
*
T0
ч
/gradients/add_6_grad/tuple/control_dependency_1Identitygradients/add_6_grad/Reshape_1&^gradients/add_6_grad/tuple/group_deps*&
_output_shapes
:
*
T0*1
_class'
%#loc:@gradients/add_6_grad/Reshape_1

gradients/conv2d_6_grad/ShapeNShapeNrelu_5Variable_12/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_6_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   
ц
+gradients/conv2d_6_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_6_grad/ShapeNVariable_12/read-gradients/add_6_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations

Й
,gradients/conv2d_6_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_5gradients/conv2d_6_grad/Const-gradients/add_6_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:





(gradients/conv2d_6_grad/tuple/group_depsNoOp,^gradients/conv2d_6_grad/Conv2DBackpropInput-^gradients/conv2d_6_grad/Conv2DBackpropFilter

0gradients/conv2d_6_grad/tuple/control_dependencyIdentity+gradients/conv2d_6_grad/Conv2DBackpropInput)^gradients/conv2d_6_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_6_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ


2gradients/conv2d_6_grad/tuple/control_dependency_1Identity,gradients/conv2d_6_grad/Conv2DBackpropFilter)^gradients/conv2d_6_grad/tuple/group_deps*&
_output_shapes
:



*
T0*?
_class5
31loc:@gradients/conv2d_6_grad/Conv2DBackpropFilter

gradients/relu_5_grad/ReluGradReluGrad0gradients/conv2d_6_grad/tuple/control_dependencyrelu_5*
T0*/
_output_shapes
:џџџџџџџџџ

b
gradients/add_5_grad/ShapeShapeconv2d_5*
T0*
out_type0*
_output_shapes
:
u
gradients/add_5_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
К
*gradients/add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_5_grad/Shapegradients/add_5_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_5_grad/SumSumgradients/relu_5_grad/ReluGrad*gradients/add_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_5_grad/ReshapeReshapegradients/add_5_grad/Sumgradients/add_5_grad/Shape*/
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0
Џ
gradients/add_5_grad/Sum_1Sumgradients/relu_5_grad/ReluGrad,gradients/add_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ђ
gradients/add_5_grad/Reshape_1Reshapegradients/add_5_grad/Sum_1gradients/add_5_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:

m
%gradients/add_5_grad/tuple/group_depsNoOp^gradients/add_5_grad/Reshape^gradients/add_5_grad/Reshape_1
ъ
-gradients/add_5_grad/tuple/control_dependencyIdentitygradients/add_5_grad/Reshape&^gradients/add_5_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*/
_class%
#!loc:@gradients/add_5_grad/Reshape
ч
/gradients/add_5_grad/tuple/control_dependency_1Identitygradients/add_5_grad/Reshape_1&^gradients/add_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_5_grad/Reshape_1*&
_output_shapes
:


gradients/conv2d_5_grad/ShapeNShapeNPool_4Variable_10/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_5_grad/ConstConst*
_output_shapes
:*%
valueB"
   
   
   
   *
dtype0
ц
+gradients/conv2d_5_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_5_grad/ShapeNVariable_10/read-gradients/add_5_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Й
,gradients/conv2d_5_grad/Conv2DBackpropFilterConv2DBackpropFilterPool_4gradients/conv2d_5_grad/Const-gradients/add_5_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0

(gradients/conv2d_5_grad/tuple/group_depsNoOp,^gradients/conv2d_5_grad/Conv2DBackpropInput-^gradients/conv2d_5_grad/Conv2DBackpropFilter

0gradients/conv2d_5_grad/tuple/control_dependencyIdentity+gradients/conv2d_5_grad/Conv2DBackpropInput)^gradients/conv2d_5_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ
*
T0*>
_class4
20loc:@gradients/conv2d_5_grad/Conv2DBackpropInput

2gradients/conv2d_5_grad/tuple/control_dependency_1Identity,gradients/conv2d_5_grad/Conv2DBackpropFilter)^gradients/conv2d_5_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_5_grad/Conv2DBackpropFilter*&
_output_shapes
:




ў
!gradients/Pool_4_grad/MaxPoolGradMaxPoolGradrelu_4Pool_40gradients/conv2d_5_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ22


gradients/relu_4_grad/ReluGradReluGrad!gradients/Pool_4_grad/MaxPoolGradrelu_4*
T0*/
_output_shapes
:џџџџџџџџџ22

b
gradients/add_4_grad/ShapeShapeconv2d_4*
out_type0*
_output_shapes
:*
T0
u
gradients/add_4_grad/Shape_1Const*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
К
*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_4_grad/SumSumgradients/relu_4_grad/ReluGrad*gradients/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ѕ
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ22

Џ
gradients/add_4_grad/Sum_1Sumgradients/relu_4_grad/ReluGrad,gradients/add_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ђ
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:22

m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
ъ
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ22
*
T0*/
_class%
#!loc:@gradients/add_4_grad/Reshape
ч
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1*&
_output_shapes
:22


gradients/conv2d_4_grad/ShapeNShapeNrelu_3Variable_8/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_4_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
х
+gradients/conv2d_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_4_grad/ShapeNVariable_8/read-gradients/add_4_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
Й
,gradients/conv2d_4_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_3gradients/conv2d_4_grad/Const-gradients/add_4_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

(gradients/conv2d_4_grad/tuple/group_depsNoOp,^gradients/conv2d_4_grad/Conv2DBackpropInput-^gradients/conv2d_4_grad/Conv2DBackpropFilter

0gradients/conv2d_4_grad/tuple/control_dependencyIdentity+gradients/conv2d_4_grad/Conv2DBackpropInput)^gradients/conv2d_4_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_4_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ22


2gradients/conv2d_4_grad/tuple/control_dependency_1Identity,gradients/conv2d_4_grad/Conv2DBackpropFilter)^gradients/conv2d_4_grad/tuple/group_deps*&
_output_shapes
:



*
T0*?
_class5
31loc:@gradients/conv2d_4_grad/Conv2DBackpropFilter

gradients/relu_3_grad/ReluGradReluGrad0gradients/conv2d_4_grad/tuple/control_dependencyrelu_3*
T0*/
_output_shapes
:џџџџџџџџџ22

b
gradients/add_3_grad/ShapeShapeconv2d_3*
T0*
out_type0*
_output_shapes
:
u
gradients/add_3_grad/Shape_1Const*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
К
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_3_grad/SumSumgradients/relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ22

Џ
gradients/add_3_grad/Sum_1Sumgradients/relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ђ
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:22

m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
ъ
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape*/
_output_shapes
:џџџџџџџџџ22

ч
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1*&
_output_shapes
:22
*
T0

gradients/conv2d_3_grad/ShapeNShapeNrelu_2Variable_6/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_3_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
х
+gradients/conv2d_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_3_grad/ShapeNVariable_6/read-gradients/add_3_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Й
,gradients/conv2d_3_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_2gradients/conv2d_3_grad/Const-gradients/add_3_grad/tuple/control_dependency*&
_output_shapes
:



*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

(gradients/conv2d_3_grad/tuple/group_depsNoOp,^gradients/conv2d_3_grad/Conv2DBackpropInput-^gradients/conv2d_3_grad/Conv2DBackpropFilter

0gradients/conv2d_3_grad/tuple/control_dependencyIdentity+gradients/conv2d_3_grad/Conv2DBackpropInput)^gradients/conv2d_3_grad/tuple/group_deps*>
_class4
20loc:@gradients/conv2d_3_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ22
*
T0

2gradients/conv2d_3_grad/tuple/control_dependency_1Identity,gradients/conv2d_3_grad/Conv2DBackpropFilter)^gradients/conv2d_3_grad/tuple/group_deps*?
_class5
31loc:@gradients/conv2d_3_grad/Conv2DBackpropFilter*&
_output_shapes
:



*
T0

gradients/relu_2_grad/ReluGradReluGrad0gradients/conv2d_3_grad/tuple/control_dependencyrelu_2*
T0*/
_output_shapes
:џџџџџџџџџ22

b
gradients/add_2_grad/ShapeShapeconv2d_2*
T0*
out_type0*
_output_shapes
:
u
gradients/add_2_grad/Shape_1Const*
_output_shapes
:*%
valueB"   2   2   
   *
dtype0
К
*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/add_2_grad/SumSumgradients/relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ѕ
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*/
_output_shapes
:џџџџџџџџџ22
*
T0*
Tshape0
Џ
gradients/add_2_grad/Sum_1Sumgradients/relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ђ
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0*&
_output_shapes
:22

m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
ъ
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape*/
_output_shapes
:џџџџџџџџџ22

ч
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1*&
_output_shapes
:22


gradients/conv2d_2_grad/ShapeNShapeNrelu_1Variable_4/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_2_grad/ConstConst*%
valueB"
   
   
   
   *
dtype0*
_output_shapes
:
х
+gradients/conv2d_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_2_grad/ShapeNVariable_4/read-gradients/add_2_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Й
,gradients/conv2d_2_grad/Conv2DBackpropFilterConv2DBackpropFilterrelu_1gradients/conv2d_2_grad/Const-gradients/add_2_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:





(gradients/conv2d_2_grad/tuple/group_depsNoOp,^gradients/conv2d_2_grad/Conv2DBackpropInput-^gradients/conv2d_2_grad/Conv2DBackpropFilter

0gradients/conv2d_2_grad/tuple/control_dependencyIdentity+gradients/conv2d_2_grad/Conv2DBackpropInput)^gradients/conv2d_2_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ22
*
T0*>
_class4
20loc:@gradients/conv2d_2_grad/Conv2DBackpropInput

2gradients/conv2d_2_grad/tuple/control_dependency_1Identity,gradients/conv2d_2_grad/Conv2DBackpropFilter)^gradients/conv2d_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_2_grad/Conv2DBackpropFilter*&
_output_shapes
:





gradients/relu_1_grad/ReluGradReluGrad0gradients/conv2d_2_grad/tuple/control_dependencyrelu_1*/
_output_shapes
:џџџџџџџџџ22
*
T0
b
gradients/add_1_grad/ShapeShapeconv2d_1*
_output_shapes
:*
T0*
out_type0
u
gradients/add_1_grad/Shape_1Const*%
valueB"   2   2   
   *
dtype0*
_output_shapes
:
К
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/add_1_grad/SumSumgradients/relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ѕ
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*/
_output_shapes
:џџџџџџџџџ22
*
T0*
Tshape0
Џ
gradients/add_1_grad/Sum_1Sumgradients/relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ђ
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*&
_output_shapes
:22
*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
ъ
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*/
_output_shapes
:џџџџџџџџџ22

ч
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*&
_output_shapes
:22


gradients/conv2d_1_grad/ShapeNShapeNPool_0Variable_2/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_1_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   
х
+gradients/conv2d_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Й
,gradients/conv2d_1_grad/Conv2DBackpropFilterConv2DBackpropFilterPool_0gradients/conv2d_1_grad/Const-gradients/add_1_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:



*
	dilations


(gradients/conv2d_1_grad/tuple/group_depsNoOp,^gradients/conv2d_1_grad/Conv2DBackpropInput-^gradients/conv2d_1_grad/Conv2DBackpropFilter

0gradients/conv2d_1_grad/tuple/control_dependencyIdentity+gradients/conv2d_1_grad/Conv2DBackpropInput)^gradients/conv2d_1_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ22
*
T0*>
_class4
20loc:@gradients/conv2d_1_grad/Conv2DBackpropInput

2gradients/conv2d_1_grad/tuple/control_dependency_1Identity,gradients/conv2d_1_grad/Conv2DBackpropFilter)^gradients/conv2d_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/conv2d_1_grad/Conv2DBackpropFilter*&
_output_shapes
:




ў
!gradients/Pool_0_grad/MaxPoolGradMaxPoolGradrelu_0Pool_00gradients/conv2d_1_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџdd


gradients/relu_0_grad/ReluGradReluGrad!gradients/Pool_0_grad/MaxPoolGradrelu_0*
T0*/
_output_shapes
:џџџџџџџџџdd

`
gradients/add_grad/ShapeShapeconv2d_0*
T0*
out_type0*
_output_shapes
:
s
gradients/add_grad/Shape_1Const*%
valueB"   d   d   
   *
dtype0*
_output_shapes
:
Д
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ї
gradients/add_grad/SumSumgradients/relu_0_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*/
_output_shapes
:џџџџџџџџџdd
*
T0*
Tshape0
Ћ
gradients/add_grad/Sum_1Sumgradients/relu_0_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*&
_output_shapes
:dd
*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
т
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџdd

п
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*&
_output_shapes
:dd


gradients/conv2d_0_grad/ShapeNShapeNConv_ReshapeVariable/read*
T0*
out_type0*
N* 
_output_shapes
::
v
gradients/conv2d_0_grad/ConstConst*%
valueB"
   
      
   *
dtype0*
_output_shapes
:
с
+gradients/conv2d_0_grad/Conv2DBackpropInputConv2DBackpropInputgradients/conv2d_0_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
Н
,gradients/conv2d_0_grad/Conv2DBackpropFilterConv2DBackpropFilterConv_Reshapegradients/conv2d_0_grad/Const+gradients/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:




(gradients/conv2d_0_grad/tuple/group_depsNoOp,^gradients/conv2d_0_grad/Conv2DBackpropInput-^gradients/conv2d_0_grad/Conv2DBackpropFilter

0gradients/conv2d_0_grad/tuple/control_dependencyIdentity+gradients/conv2d_0_grad/Conv2DBackpropInput)^gradients/conv2d_0_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/conv2d_0_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџdd

2gradients/conv2d_0_grad/tuple/control_dependency_1Identity,gradients/conv2d_0_grad/Conv2DBackpropFilter)^gradients/conv2d_0_grad/tuple/group_deps*&
_output_shapes
:


*
T0*?
_class5
31loc:@gradients/conv2d_0_grad/Conv2DBackpropFilter
{
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable*
	container 
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
_output_shapes
: *
T0
{
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wО?*
_class
loc:@Variable

beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@Variable
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
Ѕ
/Variable/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
      
   *
_class
loc:@Variable*
dtype0*
_output_shapes
:

%Variable/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable*
dtype0*
_output_shapes
: 
п
Variable/Adam/Initializer/zerosFill/Variable/Adam/Initializer/zeros/shape_as_tensor%Variable/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable*&
_output_shapes
:


*
T0
Ў
Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
	container *
shape:


*
dtype0*&
_output_shapes
:



Х
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:



{
Variable/Adam/readIdentityVariable/Adam*&
_output_shapes
:


*
T0*
_class
loc:@Variable
Ї
1Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
      
   *
_class
loc:@Variable*
dtype0*
_output_shapes
:

'Variable/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable
х
!Variable/Adam_1/Initializer/zerosFill1Variable/Adam_1/Initializer/zeros/shape_as_tensor'Variable/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable*&
_output_shapes
:



А
Variable/Adam_1
VariableV2*
	container *
shape:


*
dtype0*&
_output_shapes
:


*
shared_name *
_class
loc:@Variable
Ы
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable*
validate_shape(*&
_output_shapes
:


*
use_locking(

Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*&
_output_shapes
:



Љ
1Variable_1/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   d   d   
   *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:

'Variable_1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
ч
!Variable_1/Adam/Initializer/zerosFill1Variable_1/Adam/Initializer/zeros/shape_as_tensor'Variable_1/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_1*&
_output_shapes
:dd

В
Variable_1/Adam
VariableV2*
dtype0*&
_output_shapes
:dd
*
shared_name *
_class
loc:@Variable_1*
	container *
shape:dd

Э
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*&
_output_shapes
:dd
*
use_locking(

Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*&
_output_shapes
:dd

Ћ
3Variable_1/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   d   d   
   *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:

)Variable_1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
: 
э
#Variable_1/Adam_1/Initializer/zerosFill3Variable_1/Adam_1/Initializer/zeros/shape_as_tensor)Variable_1/Adam_1/Initializer/zeros/Const*&
_output_shapes
:dd
*
T0*

index_type0*
_class
loc:@Variable_1
Д
Variable_1/Adam_1
VariableV2*
	container *
shape:dd
*
dtype0*&
_output_shapes
:dd
*
shared_name *
_class
loc:@Variable_1
г
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:dd
*
use_locking(*
T0*
_class
loc:@Variable_1

Variable_1/Adam_1/readIdentityVariable_1/Adam_1*&
_output_shapes
:dd
*
T0*
_class
loc:@Variable_1
Љ
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_2*
dtype0*
_output_shapes
:

'Variable_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
ч
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*&
_output_shapes
:



*
T0*

index_type0*
_class
loc:@Variable_2
В
Variable_2/Adam
VariableV2*
_class
loc:@Variable_2*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name 
Э
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:





Variable_2/Adam/readIdentityVariable_2/Adam*&
_output_shapes
:



*
T0*
_class
loc:@Variable_2
Ћ
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_2*
dtype0

)Variable_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_2*
dtype0*
_output_shapes
: 
э
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_2*&
_output_shapes
:




Д
Variable_2/Adam_1
VariableV2*
_class
loc:@Variable_2*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name 
г
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:





Variable_2/Adam_1/readIdentityVariable_2/Adam_1*&
_output_shapes
:



*
T0*
_class
loc:@Variable_2
Љ
1Variable_3/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

'Variable_3/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
ч
!Variable_3/Adam/Initializer/zerosFill1Variable_3/Adam/Initializer/zeros/shape_as_tensor'Variable_3/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_3*&
_output_shapes
:22
*
T0
В
Variable_3/Adam
VariableV2*
	container *
shape:22
*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_3
Э
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_3

Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*&
_output_shapes
:22

Ћ
3Variable_3/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

)Variable_3/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
: 
э
#Variable_3/Adam_1/Initializer/zerosFill3Variable_3/Adam_1/Initializer/zeros/shape_as_tensor)Variable_3/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_3*&
_output_shapes
:22

Д
Variable_3/Adam_1
VariableV2*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_3*
	container *
shape:22
*
dtype0
г
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:22


Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*&
_output_shapes
:22

Љ
1Variable_4/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_4*
dtype0*
_output_shapes
:

'Variable_4/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
ч
!Variable_4/Adam/Initializer/zerosFill1Variable_4/Adam/Initializer/zeros/shape_as_tensor'Variable_4/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_4*&
_output_shapes
:




В
Variable_4/Adam
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_4*
	container *
shape:




Э
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:





Variable_4/Adam/readIdentityVariable_4/Adam*&
_output_shapes
:



*
T0*
_class
loc:@Variable_4
Ћ
3Variable_4/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_4*
dtype0*
_output_shapes
:

)Variable_4/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_4*
dtype0*
_output_shapes
: 
э
#Variable_4/Adam_1/Initializer/zerosFill3Variable_4/Adam_1/Initializer/zeros/shape_as_tensor)Variable_4/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_4*&
_output_shapes
:




Д
Variable_4/Adam_1
VariableV2*
_class
loc:@Variable_4*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name 
г
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:





Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*&
_output_shapes
:




Љ
1Variable_5/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:

'Variable_5/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
ч
!Variable_5/Adam/Initializer/zerosFill1Variable_5/Adam/Initializer/zeros/shape_as_tensor'Variable_5/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_5*&
_output_shapes
:22

В
Variable_5/Adam
VariableV2*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_5*
	container *
shape:22
*
dtype0
Э
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
:22


Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*&
_output_shapes
:22

Ћ
3Variable_5/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:

)Variable_5/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
: 
э
#Variable_5/Adam_1/Initializer/zerosFill3Variable_5/Adam_1/Initializer/zeros/shape_as_tensor)Variable_5/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_5*&
_output_shapes
:22
*
T0
Д
Variable_5/Adam_1
VariableV2*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_5*
	container *
shape:22

г
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_5

Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*&
_output_shapes
:22
*
T0
Љ
1Variable_6/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_6*
dtype0*
_output_shapes
:

'Variable_6/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_6*
dtype0*
_output_shapes
: 
ч
!Variable_6/Adam/Initializer/zerosFill1Variable_6/Adam/Initializer/zeros/shape_as_tensor'Variable_6/Adam/Initializer/zeros/Const*&
_output_shapes
:



*
T0*

index_type0*
_class
loc:@Variable_6
В
Variable_6/Adam
VariableV2*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_6
Э
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(

Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*&
_output_shapes
:




Ћ
3Variable_6/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_6

)Variable_6/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_6*
dtype0*
_output_shapes
: 
э
#Variable_6/Adam_1/Initializer/zerosFill3Variable_6/Adam_1/Initializer/zeros/shape_as_tensor)Variable_6/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_6*&
_output_shapes
:




Д
Variable_6/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_6*
	container *
shape:



*
dtype0*&
_output_shapes
:




г
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(

Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*&
_output_shapes
:




Љ
1Variable_7/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

'Variable_7/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_7*
dtype0
ч
!Variable_7/Adam/Initializer/zerosFill1Variable_7/Adam/Initializer/zeros/shape_as_tensor'Variable_7/Adam/Initializer/zeros/Const*&
_output_shapes
:22
*
T0*

index_type0*
_class
loc:@Variable_7
В
Variable_7/Adam
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:22
*
dtype0*&
_output_shapes
:22

Э
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(*&
_output_shapes
:22


Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*&
_output_shapes
:22

Ћ
3Variable_7/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:

)Variable_7/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
: 
э
#Variable_7/Adam_1/Initializer/zerosFill3Variable_7/Adam_1/Initializer/zeros/shape_as_tensor)Variable_7/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_7*&
_output_shapes
:22

Д
Variable_7/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
	container *
shape:22
*
dtype0*&
_output_shapes
:22

г
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_7*
validate_shape(

Variable_7/Adam_1/readIdentityVariable_7/Adam_1*&
_output_shapes
:22
*
T0*
_class
loc:@Variable_7
Љ
1Variable_8/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_8*
dtype0*
_output_shapes
:

'Variable_8/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_8*
dtype0*
_output_shapes
: 
ч
!Variable_8/Adam/Initializer/zerosFill1Variable_8/Adam/Initializer/zeros/shape_as_tensor'Variable_8/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_8*&
_output_shapes
:



*
T0
В
Variable_8/Adam
VariableV2*
shared_name *
_class
loc:@Variable_8*
	container *
shape:



*
dtype0*&
_output_shapes
:




Э
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(

Variable_8/Adam/readIdentityVariable_8/Adam*&
_output_shapes
:



*
T0*
_class
loc:@Variable_8
Ћ
3Variable_8/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_8*
dtype0

)Variable_8/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_8*
dtype0
э
#Variable_8/Adam_1/Initializer/zerosFill3Variable_8/Adam_1/Initializer/zeros/shape_as_tensor)Variable_8/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_8*&
_output_shapes
:




Д
Variable_8/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_8*
	container *
shape:



*
dtype0*&
_output_shapes
:




г
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*&
_output_shapes
:





Variable_8/Adam_1/readIdentityVariable_8/Adam_1*&
_output_shapes
:



*
T0*
_class
loc:@Variable_8
Љ
1Variable_9/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_9*
dtype0*
_output_shapes
:

'Variable_9/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
: 
ч
!Variable_9/Adam/Initializer/zerosFill1Variable_9/Adam/Initializer/zeros/shape_as_tensor'Variable_9/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_9*&
_output_shapes
:22

В
Variable_9/Adam
VariableV2*
	container *
shape:22
*
dtype0*&
_output_shapes
:22
*
shared_name *
_class
loc:@Variable_9
Э
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*&
_output_shapes
:22


Variable_9/Adam/readIdentityVariable_9/Adam*&
_output_shapes
:22
*
T0*
_class
loc:@Variable_9
Ћ
3Variable_9/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"   2   2   
   *
_class
loc:@Variable_9*
dtype0*
_output_shapes
:

)Variable_9/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
: 
э
#Variable_9/Adam_1/Initializer/zerosFill3Variable_9/Adam_1/Initializer/zeros/shape_as_tensor)Variable_9/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_9*&
_output_shapes
:22
*
T0
Д
Variable_9/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_9*
	container *
shape:22
*
dtype0*&
_output_shapes
:22

г
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_9*
validate_shape(*&
_output_shapes
:22


Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_class
loc:@Variable_9*&
_output_shapes
:22

Ћ
2Variable_10/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_10*
dtype0*
_output_shapes
:

(Variable_10/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_10*
dtype0*
_output_shapes
: 
ы
"Variable_10/Adam/Initializer/zerosFill2Variable_10/Adam/Initializer/zeros/shape_as_tensor(Variable_10/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_10*&
_output_shapes
:



*
T0
Д
Variable_10/Adam
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_10*
	container *
shape:




б
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(

Variable_10/Adam/readIdentityVariable_10/Adam*
T0*
_class
loc:@Variable_10*&
_output_shapes
:




­
4Variable_10/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_10*
dtype0*
_output_shapes
:

*Variable_10/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_10*
dtype0*
_output_shapes
: 
ё
$Variable_10/Adam_1/Initializer/zerosFill4Variable_10/Adam_1/Initializer/zeros/shape_as_tensor*Variable_10/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_10*&
_output_shapes
:



*
T0
Ж
Variable_10/Adam_1
VariableV2*
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_10*
	container 
з
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
_class
loc:@Variable_10*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0

Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*
_class
loc:@Variable_10*&
_output_shapes
:




Ћ
2Variable_11/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_11*
dtype0*
_output_shapes
:

(Variable_11/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_11
ы
"Variable_11/Adam/Initializer/zerosFill2Variable_11/Adam/Initializer/zeros/shape_as_tensor(Variable_11/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_11*&
_output_shapes
:
*
T0
Д
Variable_11/Adam
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_11*
	container *
shape:

б
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_11

Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_class
loc:@Variable_11*&
_output_shapes
:

­
4Variable_11/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_11*
dtype0*
_output_shapes
:

*Variable_11/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_11*
dtype0
ё
$Variable_11/Adam_1/Initializer/zerosFill4Variable_11/Adam_1/Initializer/zeros/shape_as_tensor*Variable_11/Adam_1/Initializer/zeros/Const*&
_output_shapes
:
*
T0*

index_type0*
_class
loc:@Variable_11
Ж
Variable_11/Adam_1
VariableV2*
shape:
*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_11*
	container 
з
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*&
_output_shapes
:


Variable_11/Adam_1/readIdentityVariable_11/Adam_1*&
_output_shapes
:
*
T0*
_class
loc:@Variable_11
Ћ
2Variable_12/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_12*
dtype0*
_output_shapes
:

(Variable_12/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_12*
dtype0*
_output_shapes
: 
ы
"Variable_12/Adam/Initializer/zerosFill2Variable_12/Adam/Initializer/zeros/shape_as_tensor(Variable_12/Adam/Initializer/zeros/Const*&
_output_shapes
:



*
T0*

index_type0*
_class
loc:@Variable_12
Д
Variable_12/Adam
VariableV2*
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_12*
	container 
б
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
_class
loc:@Variable_12*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0

Variable_12/Adam/readIdentityVariable_12/Adam*&
_output_shapes
:



*
T0*
_class
loc:@Variable_12
­
4Variable_12/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_12*
dtype0*
_output_shapes
:

*Variable_12/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_12*
dtype0*
_output_shapes
: 
ё
$Variable_12/Adam_1/Initializer/zerosFill4Variable_12/Adam_1/Initializer/zeros/shape_as_tensor*Variable_12/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_12*&
_output_shapes
:




Ж
Variable_12/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_12*
	container *
shape:



*
dtype0*&
_output_shapes
:




з
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*&
_output_shapes
:





Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_class
loc:@Variable_12*&
_output_shapes
:




Ћ
2Variable_13/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         
   *
_class
loc:@Variable_13

(Variable_13/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_13
ы
"Variable_13/Adam/Initializer/zerosFill2Variable_13/Adam/Initializer/zeros/shape_as_tensor(Variable_13/Adam/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_13*&
_output_shapes
:
*
T0
Д
Variable_13/Adam
VariableV2*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_13
б
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_13

Variable_13/Adam/readIdentityVariable_13/Adam*&
_output_shapes
:
*
T0*
_class
loc:@Variable_13
­
4Variable_13/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_13*
dtype0*
_output_shapes
:

*Variable_13/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
: 
ё
$Variable_13/Adam_1/Initializer/zerosFill4Variable_13/Adam_1/Initializer/zeros/shape_as_tensor*Variable_13/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_13*&
_output_shapes
:

Ж
Variable_13/Adam_1
VariableV2*
_class
loc:@Variable_13*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name 
з
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_13

Variable_13/Adam_1/readIdentityVariable_13/Adam_1*&
_output_shapes
:
*
T0*
_class
loc:@Variable_13
Ћ
2Variable_14/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_14*
dtype0

(Variable_14/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_14*
dtype0*
_output_shapes
: 
ы
"Variable_14/Adam/Initializer/zerosFill2Variable_14/Adam/Initializer/zeros/shape_as_tensor(Variable_14/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_14*&
_output_shapes
:




Д
Variable_14/Adam
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_14*
	container *
shape:




б
Variable_14/Adam/AssignAssignVariable_14/Adam"Variable_14/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(*&
_output_shapes
:





Variable_14/Adam/readIdentityVariable_14/Adam*
T0*
_class
loc:@Variable_14*&
_output_shapes
:




­
4Variable_14/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_14*
dtype0*
_output_shapes
:

*Variable_14/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_14*
dtype0
ё
$Variable_14/Adam_1/Initializer/zerosFill4Variable_14/Adam_1/Initializer/zeros/shape_as_tensor*Variable_14/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_14*&
_output_shapes
:




Ж
Variable_14/Adam_1
VariableV2*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_14
з
Variable_14/Adam_1/AssignAssignVariable_14/Adam_1$Variable_14/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_14*
validate_shape(*&
_output_shapes
:



*
use_locking(

Variable_14/Adam_1/readIdentityVariable_14/Adam_1*&
_output_shapes
:



*
T0*
_class
loc:@Variable_14
Ћ
2Variable_15/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

(Variable_15/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
: 
ы
"Variable_15/Adam/Initializer/zerosFill2Variable_15/Adam/Initializer/zeros/shape_as_tensor(Variable_15/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_15*&
_output_shapes
:

Д
Variable_15/Adam
VariableV2*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_15
б
Variable_15/Adam/AssignAssignVariable_15/Adam"Variable_15/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_15*
validate_shape(*&
_output_shapes
:
*
use_locking(

Variable_15/Adam/readIdentityVariable_15/Adam*
T0*
_class
loc:@Variable_15*&
_output_shapes
:

­
4Variable_15/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

*Variable_15/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
: 
ё
$Variable_15/Adam_1/Initializer/zerosFill4Variable_15/Adam_1/Initializer/zeros/shape_as_tensor*Variable_15/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_15*&
_output_shapes
:

Ж
Variable_15/Adam_1
VariableV2*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_15*
	container *
shape:
*
dtype0
з
Variable_15/Adam_1/AssignAssignVariable_15/Adam_1$Variable_15/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*&
_output_shapes
:


Variable_15/Adam_1/readIdentityVariable_15/Adam_1*&
_output_shapes
:
*
T0*
_class
loc:@Variable_15
Ћ
2Variable_16/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_16*
dtype0

(Variable_16/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_16*
dtype0*
_output_shapes
: 
ы
"Variable_16/Adam/Initializer/zerosFill2Variable_16/Adam/Initializer/zeros/shape_as_tensor(Variable_16/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_16*&
_output_shapes
:




Д
Variable_16/Adam
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_16*
	container *
shape:




б
Variable_16/Adam/AssignAssignVariable_16/Adam"Variable_16/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*&
_output_shapes
:





Variable_16/Adam/readIdentityVariable_16/Adam*
_class
loc:@Variable_16*&
_output_shapes
:



*
T0
­
4Variable_16/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_16*
dtype0*
_output_shapes
:

*Variable_16/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_16*
dtype0
ё
$Variable_16/Adam_1/Initializer/zerosFill4Variable_16/Adam_1/Initializer/zeros/shape_as_tensor*Variable_16/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_16*&
_output_shapes
:




Ж
Variable_16/Adam_1
VariableV2*
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_16*
	container 
з
Variable_16/Adam_1/AssignAssignVariable_16/Adam_1$Variable_16/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*&
_output_shapes
:





Variable_16/Adam_1/readIdentityVariable_16/Adam_1*&
_output_shapes
:



*
T0*
_class
loc:@Variable_16
Ћ
2Variable_17/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_17*
dtype0*
_output_shapes
:

(Variable_17/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_17*
dtype0*
_output_shapes
: 
ы
"Variable_17/Adam/Initializer/zerosFill2Variable_17/Adam/Initializer/zeros/shape_as_tensor(Variable_17/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_17*&
_output_shapes
:

Д
Variable_17/Adam
VariableV2*
_class
loc:@Variable_17*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name 
б
Variable_17/Adam/AssignAssignVariable_17/Adam"Variable_17/Adam/Initializer/zeros*
_class
loc:@Variable_17*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0

Variable_17/Adam/readIdentityVariable_17/Adam*
T0*
_class
loc:@Variable_17*&
_output_shapes
:

­
4Variable_17/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_17*
dtype0*
_output_shapes
:

*Variable_17/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_17*
dtype0*
_output_shapes
: 
ё
$Variable_17/Adam_1/Initializer/zerosFill4Variable_17/Adam_1/Initializer/zeros/shape_as_tensor*Variable_17/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_17*&
_output_shapes
:
*
T0
Ж
Variable_17/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_17*
	container *
shape:
*
dtype0*&
_output_shapes
:

з
Variable_17/Adam_1/AssignAssignVariable_17/Adam_1$Variable_17/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_17

Variable_17/Adam_1/readIdentityVariable_17/Adam_1*
T0*
_class
loc:@Variable_17*&
_output_shapes
:

Ћ
2Variable_18/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_18*
dtype0*
_output_shapes
:

(Variable_18/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_18*
dtype0*
_output_shapes
: 
ы
"Variable_18/Adam/Initializer/zerosFill2Variable_18/Adam/Initializer/zeros/shape_as_tensor(Variable_18/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_18*&
_output_shapes
:




Д
Variable_18/Adam
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_18*
	container *
shape:




б
Variable_18/Adam/AssignAssignVariable_18/Adam"Variable_18/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(*&
_output_shapes
:





Variable_18/Adam/readIdentityVariable_18/Adam*
T0*
_class
loc:@Variable_18*&
_output_shapes
:




­
4Variable_18/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_18*
dtype0*
_output_shapes
:

*Variable_18/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_18*
dtype0*
_output_shapes
: 
ё
$Variable_18/Adam_1/Initializer/zerosFill4Variable_18/Adam_1/Initializer/zeros/shape_as_tensor*Variable_18/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_18*&
_output_shapes
:



*
T0
Ж
Variable_18/Adam_1
VariableV2*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_18
з
Variable_18/Adam_1/AssignAssignVariable_18/Adam_1$Variable_18/Adam_1/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(

Variable_18/Adam_1/readIdentityVariable_18/Adam_1*
T0*
_class
loc:@Variable_18*&
_output_shapes
:




Ћ
2Variable_19/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_19*
dtype0*
_output_shapes
:

(Variable_19/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_19
ы
"Variable_19/Adam/Initializer/zerosFill2Variable_19/Adam/Initializer/zeros/shape_as_tensor(Variable_19/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_19*&
_output_shapes
:

Д
Variable_19/Adam
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_19*
	container *
shape:

б
Variable_19/Adam/AssignAssignVariable_19/Adam"Variable_19/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(*&
_output_shapes
:


Variable_19/Adam/readIdentityVariable_19/Adam*&
_output_shapes
:
*
T0*
_class
loc:@Variable_19
­
4Variable_19/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_19*
dtype0*
_output_shapes
:

*Variable_19/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_19*
dtype0*
_output_shapes
: 
ё
$Variable_19/Adam_1/Initializer/zerosFill4Variable_19/Adam_1/Initializer/zeros/shape_as_tensor*Variable_19/Adam_1/Initializer/zeros/Const*&
_output_shapes
:
*
T0*

index_type0*
_class
loc:@Variable_19
Ж
Variable_19/Adam_1
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_19*
	container *
shape:

з
Variable_19/Adam_1/AssignAssignVariable_19/Adam_1$Variable_19/Adam_1/Initializer/zeros*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_19*
validate_shape(

Variable_19/Adam_1/readIdentityVariable_19/Adam_1*
T0*
_class
loc:@Variable_19*&
_output_shapes
:

Ћ
2Variable_20/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*%
valueB"
   
   
   
   *
_class
loc:@Variable_20*
dtype0

(Variable_20/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 
ы
"Variable_20/Adam/Initializer/zerosFill2Variable_20/Adam/Initializer/zeros/shape_as_tensor(Variable_20/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_20*&
_output_shapes
:




Д
Variable_20/Adam
VariableV2*
_class
loc:@Variable_20*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name 
б
Variable_20/Adam/AssignAssignVariable_20/Adam"Variable_20/Adam/Initializer/zeros*
_class
loc:@Variable_20*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0

Variable_20/Adam/readIdentityVariable_20/Adam*
_class
loc:@Variable_20*&
_output_shapes
:



*
T0
­
4Variable_20/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_20*
dtype0*
_output_shapes
:

*Variable_20/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_20*
dtype0*
_output_shapes
: 
ё
$Variable_20/Adam_1/Initializer/zerosFill4Variable_20/Adam_1/Initializer/zeros/shape_as_tensor*Variable_20/Adam_1/Initializer/zeros/Const*

index_type0*
_class
loc:@Variable_20*&
_output_shapes
:



*
T0
Ж
Variable_20/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_20*
	container *
shape:



*
dtype0*&
_output_shapes
:




з
Variable_20/Adam_1/AssignAssignVariable_20/Adam_1$Variable_20/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*&
_output_shapes
:





Variable_20/Adam_1/readIdentityVariable_20/Adam_1*&
_output_shapes
:



*
T0*
_class
loc:@Variable_20
Ћ
2Variable_21/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_21*
dtype0*
_output_shapes
:

(Variable_21/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_21*
dtype0*
_output_shapes
: 
ы
"Variable_21/Adam/Initializer/zerosFill2Variable_21/Adam/Initializer/zeros/shape_as_tensor(Variable_21/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_21*&
_output_shapes
:

Д
Variable_21/Adam
VariableV2*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_21*
	container *
shape:
*
dtype0
б
Variable_21/Adam/AssignAssignVariable_21/Adam"Variable_21/Adam/Initializer/zeros*
_class
loc:@Variable_21*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0

Variable_21/Adam/readIdentityVariable_21/Adam*
T0*
_class
loc:@Variable_21*&
_output_shapes
:

­
4Variable_21/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_21*
dtype0*
_output_shapes
:

*Variable_21/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_21*
dtype0
ё
$Variable_21/Adam_1/Initializer/zerosFill4Variable_21/Adam_1/Initializer/zeros/shape_as_tensor*Variable_21/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_21*&
_output_shapes
:

Ж
Variable_21/Adam_1
VariableV2*
_class
loc:@Variable_21*
	container *
shape:
*
dtype0*&
_output_shapes
:
*
shared_name 
з
Variable_21/Adam_1/AssignAssignVariable_21/Adam_1$Variable_21/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_21*
validate_shape(*&
_output_shapes
:
*
use_locking(

Variable_21/Adam_1/readIdentityVariable_21/Adam_1*
T0*
_class
loc:@Variable_21*&
_output_shapes
:

Ћ
2Variable_22/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_22*
dtype0*
_output_shapes
:

(Variable_22/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_22*
dtype0*
_output_shapes
: 
ы
"Variable_22/Adam/Initializer/zerosFill2Variable_22/Adam/Initializer/zeros/shape_as_tensor(Variable_22/Adam/Initializer/zeros/Const*&
_output_shapes
:



*
T0*

index_type0*
_class
loc:@Variable_22
Д
Variable_22/Adam
VariableV2*
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_22*
	container 
б
Variable_22/Adam/AssignAssignVariable_22/Adam"Variable_22/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
:



*
use_locking(

Variable_22/Adam/readIdentityVariable_22/Adam*
T0*
_class
loc:@Variable_22*&
_output_shapes
:




­
4Variable_22/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_22*
dtype0*
_output_shapes
:

*Variable_22/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_22*
dtype0*
_output_shapes
: 
ё
$Variable_22/Adam_1/Initializer/zerosFill4Variable_22/Adam_1/Initializer/zeros/shape_as_tensor*Variable_22/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_22*&
_output_shapes
:




Ж
Variable_22/Adam_1
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_22*
	container *
shape:




з
Variable_22/Adam_1/AssignAssignVariable_22/Adam_1$Variable_22/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
:





Variable_22/Adam_1/readIdentityVariable_22/Adam_1*
T0*
_class
loc:@Variable_22*&
_output_shapes
:




Ћ
2Variable_23/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         
   *
_class
loc:@Variable_23

(Variable_23/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_23*
dtype0*
_output_shapes
: 
ы
"Variable_23/Adam/Initializer/zerosFill2Variable_23/Adam/Initializer/zeros/shape_as_tensor(Variable_23/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_23*&
_output_shapes
:

Д
Variable_23/Adam
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_23*
	container *
shape:

б
Variable_23/Adam/AssignAssignVariable_23/Adam"Variable_23/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_23*
validate_shape(*&
_output_shapes
:
*
use_locking(

Variable_23/Adam/readIdentityVariable_23/Adam*
T0*
_class
loc:@Variable_23*&
_output_shapes
:

­
4Variable_23/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         
   *
_class
loc:@Variable_23

*Variable_23/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_23*
dtype0*
_output_shapes
: 
ё
$Variable_23/Adam_1/Initializer/zerosFill4Variable_23/Adam_1/Initializer/zeros/shape_as_tensor*Variable_23/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_23*&
_output_shapes
:

Ж
Variable_23/Adam_1
VariableV2*
shape:
*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_23*
	container 
з
Variable_23/Adam_1/AssignAssignVariable_23/Adam_1$Variable_23/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_23*
validate_shape(*&
_output_shapes
:


Variable_23/Adam_1/readIdentityVariable_23/Adam_1*
T0*
_class
loc:@Variable_23*&
_output_shapes
:

Ћ
2Variable_24/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_24*
dtype0*
_output_shapes
:

(Variable_24/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_24
ы
"Variable_24/Adam/Initializer/zerosFill2Variable_24/Adam/Initializer/zeros/shape_as_tensor(Variable_24/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_24*&
_output_shapes
:




Д
Variable_24/Adam
VariableV2*
	container *
shape:



*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_24
б
Variable_24/Adam/AssignAssignVariable_24/Adam"Variable_24/Adam/Initializer/zeros*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(

Variable_24/Adam/readIdentityVariable_24/Adam*
T0*
_class
loc:@Variable_24*&
_output_shapes
:




­
4Variable_24/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"
   
   
   
   *
_class
loc:@Variable_24*
dtype0*
_output_shapes
:

*Variable_24/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_24
ё
$Variable_24/Adam_1/Initializer/zerosFill4Variable_24/Adam_1/Initializer/zeros/shape_as_tensor*Variable_24/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_24*&
_output_shapes
:




Ж
Variable_24/Adam_1
VariableV2*
dtype0*&
_output_shapes
:



*
shared_name *
_class
loc:@Variable_24*
	container *
shape:




з
Variable_24/Adam_1/AssignAssignVariable_24/Adam_1$Variable_24/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(*&
_output_shapes
:





Variable_24/Adam_1/readIdentityVariable_24/Adam_1*&
_output_shapes
:



*
T0*
_class
loc:@Variable_24
Ћ
2Variable_25/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"         
   *
_class
loc:@Variable_25*
dtype0*
_output_shapes
:

(Variable_25/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_25*
dtype0*
_output_shapes
: 
ы
"Variable_25/Adam/Initializer/zerosFill2Variable_25/Adam/Initializer/zeros/shape_as_tensor(Variable_25/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_25*&
_output_shapes
:

Д
Variable_25/Adam
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_25*
	container *
shape:

б
Variable_25/Adam/AssignAssignVariable_25/Adam"Variable_25/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(*&
_output_shapes
:


Variable_25/Adam/readIdentityVariable_25/Adam*
T0*
_class
loc:@Variable_25*&
_output_shapes
:

­
4Variable_25/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"         
   *
_class
loc:@Variable_25

*Variable_25/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_25*
dtype0*
_output_shapes
: 
ё
$Variable_25/Adam_1/Initializer/zerosFill4Variable_25/Adam_1/Initializer/zeros/shape_as_tensor*Variable_25/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_25*&
_output_shapes
:

Ж
Variable_25/Adam_1
VariableV2*
dtype0*&
_output_shapes
:
*
shared_name *
_class
loc:@Variable_25*
	container *
shape:

з
Variable_25/Adam_1/AssignAssignVariable_25/Adam_1$Variable_25/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(*&
_output_shapes
:


Variable_25/Adam_1/readIdentityVariable_25/Adam_1*
_class
loc:@Variable_25*&
_output_shapes
:
*
T0
Ѓ
2Variable_26/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"ъ  	   *
_class
loc:@Variable_26*
dtype0*
_output_shapes
:

(Variable_26/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_26
ф
"Variable_26/Adam/Initializer/zerosFill2Variable_26/Adam/Initializer/zeros/shape_as_tensor(Variable_26/Adam/Initializer/zeros/Const*
_output_shapes
:	ъ	*
T0*

index_type0*
_class
loc:@Variable_26
І
Variable_26/Adam
VariableV2*
shared_name *
_class
loc:@Variable_26*
	container *
shape:	ъ	*
dtype0*
_output_shapes
:	ъ	
Ъ
Variable_26/Adam/AssignAssignVariable_26/Adam"Variable_26/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@Variable_26*
validate_shape(*
_output_shapes
:	ъ	
}
Variable_26/Adam/readIdentityVariable_26/Adam*
_output_shapes
:	ъ	*
T0*
_class
loc:@Variable_26
Ѕ
4Variable_26/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"ъ  	   *
_class
loc:@Variable_26*
dtype0

*Variable_26/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_26*
dtype0*
_output_shapes
: 
ъ
$Variable_26/Adam_1/Initializer/zerosFill4Variable_26/Adam_1/Initializer/zeros/shape_as_tensor*Variable_26/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_26*
_output_shapes
:	ъ	
Ј
Variable_26/Adam_1
VariableV2*
	container *
shape:	ъ	*
dtype0*
_output_shapes
:	ъ	*
shared_name *
_class
loc:@Variable_26
а
Variable_26/Adam_1/AssignAssignVariable_26/Adam_1$Variable_26/Adam_1/Initializer/zeros*
_class
loc:@Variable_26*
validate_shape(*
_output_shapes
:	ъ	*
use_locking(*
T0

Variable_26/Adam_1/readIdentityVariable_26/Adam_1*
T0*
_class
loc:@Variable_26*
_output_shapes
:	ъ	
Ѓ
2Variable_27/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"   	   *
_class
loc:@Variable_27*
dtype0

(Variable_27/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *
_class
loc:@Variable_27
у
"Variable_27/Adam/Initializer/zerosFill2Variable_27/Adam/Initializer/zeros/shape_as_tensor(Variable_27/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@Variable_27*
_output_shapes

:	
Є
Variable_27/Adam
VariableV2*
shared_name *
_class
loc:@Variable_27*
	container *
shape
:	*
dtype0*
_output_shapes

:	
Щ
Variable_27/Adam/AssignAssignVariable_27/Adam"Variable_27/Adam/Initializer/zeros*
_class
loc:@Variable_27*
validate_shape(*
_output_shapes

:	*
use_locking(*
T0
|
Variable_27/Adam/readIdentityVariable_27/Adam*
T0*
_class
loc:@Variable_27*
_output_shapes

:	
Ѕ
4Variable_27/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   	   *
_class
loc:@Variable_27*
dtype0*
_output_shapes
:

*Variable_27/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@Variable_27*
dtype0*
_output_shapes
: 
щ
$Variable_27/Adam_1/Initializer/zerosFill4Variable_27/Adam_1/Initializer/zeros/shape_as_tensor*Variable_27/Adam_1/Initializer/zeros/Const*
_output_shapes

:	*
T0*

index_type0*
_class
loc:@Variable_27
І
Variable_27/Adam_1
VariableV2*
	container *
shape
:	*
dtype0*
_output_shapes

:	*
shared_name *
_class
loc:@Variable_27
Я
Variable_27/Adam_1/AssignAssignVariable_27/Adam_1$Variable_27/Adam_1/Initializer/zeros*
_output_shapes

:	*
use_locking(*
T0*
_class
loc:@Variable_27*
validate_shape(

Variable_27/Adam_1/readIdentityVariable_27/Adam_1*
_output_shapes

:	*
T0*
_class
loc:@Variable_27
W
Adam/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *wО?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *wЬ+2*
dtype0
м
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_0_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable*
use_nesterov( *&
_output_shapes
:



с
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_1*
use_nesterov( *&
_output_shapes
:dd

ц
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_2*
use_nesterov( *&
_output_shapes
:




у
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*&
_output_shapes
:22
*
use_locking( *
T0*
_class
loc:@Variable_3*
use_nesterov( 
ц
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_2_grad/tuple/control_dependency_1*&
_output_shapes
:



*
use_locking( *
T0*
_class
loc:@Variable_4*
use_nesterov( 
у
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_5*
use_nesterov( *&
_output_shapes
:22

ц
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_6*
use_nesterov( *&
_output_shapes
:




у
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_7*
use_nesterov( *&
_output_shapes
:22

ц
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_4_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_8*
use_nesterov( *&
_output_shapes
:




у
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_9*
use_nesterov( *&
_output_shapes
:22
*
use_locking( 
ы
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_5_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_10*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_5_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_11*
use_nesterov( *&
_output_shapes
:

ы
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_6_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_12*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_6_grad/tuple/control_dependency_1*
_class
loc:@Variable_13*
use_nesterov( *&
_output_shapes
:
*
use_locking( *
T0
ы
!Adam/update_Variable_14/ApplyAdam	ApplyAdamVariable_14Variable_14/AdamVariable_14/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_7_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_14*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_15/ApplyAdam	ApplyAdamVariable_15Variable_15/AdamVariable_15/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_7_grad/tuple/control_dependency_1*&
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@Variable_15*
use_nesterov( 
ы
!Adam/update_Variable_16/ApplyAdam	ApplyAdamVariable_16Variable_16/AdamVariable_16/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_8_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_16*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_17/ApplyAdam	ApplyAdamVariable_17Variable_17/AdamVariable_17/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_8_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_17*
use_nesterov( *&
_output_shapes
:

ы
!Adam/update_Variable_18/ApplyAdam	ApplyAdamVariable_18Variable_18/AdamVariable_18/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/conv2d_9_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_18*
use_nesterov( *&
_output_shapes
:




ш
!Adam/update_Variable_19/ApplyAdam	ApplyAdamVariable_19Variable_19/AdamVariable_19/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_9_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_19*
use_nesterov( *&
_output_shapes
:
*
use_locking( 
ь
!Adam/update_Variable_20/ApplyAdam	ApplyAdamVariable_20Variable_20/AdamVariable_20/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/conv2d_10_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:



*
use_locking( *
T0*
_class
loc:@Variable_20
щ
!Adam/update_Variable_21/ApplyAdam	ApplyAdamVariable_21Variable_21/AdamVariable_21/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_10_grad/tuple/control_dependency_1*&
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@Variable_21*
use_nesterov( 
ь
!Adam/update_Variable_22/ApplyAdam	ApplyAdamVariable_22Variable_22/AdamVariable_22/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/conv2d_11_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_22*
use_nesterov( *&
_output_shapes
:




щ
!Adam/update_Variable_23/ApplyAdam	ApplyAdamVariable_23Variable_23/AdamVariable_23/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_11_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_23*
use_nesterov( *&
_output_shapes
:

ь
!Adam/update_Variable_24/ApplyAdam	ApplyAdamVariable_24Variable_24/AdamVariable_24/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/conv2d_12_grad/tuple/control_dependency_1*&
_output_shapes
:



*
use_locking( *
T0*
_class
loc:@Variable_24*
use_nesterov( 
щ
!Adam/update_Variable_25/ApplyAdam	ApplyAdamVariable_25Variable_25/AdamVariable_25/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_12_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_25*
use_nesterov( *&
_output_shapes
:

х
!Adam/update_Variable_26/ApplyAdam	ApplyAdamVariable_26Variable_26/AdamVariable_26/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/FC_MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_26*
use_nesterov( *
_output_shapes
:	ъ	
с
!Adam/update_Variable_27/ApplyAdam	ApplyAdamVariable_27Variable_27/AdamVariable_27/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_13_grad/tuple/control_dependency_1*
_output_shapes

:	*
use_locking( *
T0*
_class
loc:@Variable_27*
use_nesterov( 
Я
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam"^Adam/update_Variable_26/ApplyAdam"^Adam/update_Variable_27/ApplyAdam*
_class
loc:@Variable*
_output_shapes
: *
T0

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 
б

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam"^Adam/update_Variable_26/ApplyAdam"^Adam/update_Variable_27/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes
: 

AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam"^Adam/update_Variable_26/ApplyAdam"^Adam/update_Variable_27/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Т
save/SaveV2/tensor_namesConst*ѕ
valueыBшBVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15BVariable_16BVariable_17BVariable_18BVariable_19B
Variable_2BVariable_20BVariable_21BVariable_22BVariable_23BVariable_24BVariable_25BVariable_26BVariable_27B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
Variable_8B
Variable_9*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*
_output_shapes
:*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
х
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1Variable_10Variable_11Variable_12Variable_13Variable_14Variable_15Variable_16Variable_17Variable_18Variable_19
Variable_2Variable_20Variable_21Variable_22Variable_23Variable_24Variable_25Variable_26Variable_27
Variable_3
Variable_4
Variable_5
Variable_6
Variable_7
Variable_8
Variable_9**
dtypes 
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
д
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*ѕ
valueыBшBVariableB
Variable_1BVariable_10BVariable_11BVariable_12BVariable_13BVariable_14BVariable_15BVariable_16BVariable_17BVariable_18BVariable_19B
Variable_2BVariable_20BVariable_21BVariable_22BVariable_23BVariable_24BVariable_25BVariable_26BVariable_27B
Variable_3B
Variable_4B
Variable_5B
Variable_6B
Variable_7B
Variable_8B
Variable_9*
dtype0
­
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ї
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0**
dtypes 
2*
_output_shapesr
p::::::::::::::::::::::::::::
І
save/AssignAssignVariablesave/RestoreV2*
validate_shape(*&
_output_shapes
:


*
use_locking(*
T0*
_class
loc:@Variable
Ў
save/Assign_1Assign
Variable_1save/RestoreV2:1*
validate_shape(*&
_output_shapes
:dd
*
use_locking(*
T0*
_class
loc:@Variable_1
А
save/Assign_2AssignVariable_10save/RestoreV2:2*
_class
loc:@Variable_10*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
А
save/Assign_3AssignVariable_11save/RestoreV2:3*
use_locking(*
T0*
_class
loc:@Variable_11*
validate_shape(*&
_output_shapes
:

А
save/Assign_4AssignVariable_12save/RestoreV2:4*
use_locking(*
T0*
_class
loc:@Variable_12*
validate_shape(*&
_output_shapes
:




А
save/Assign_5AssignVariable_13save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@Variable_13*
validate_shape(*&
_output_shapes
:

А
save/Assign_6AssignVariable_14save/RestoreV2:6*
T0*
_class
loc:@Variable_14*
validate_shape(*&
_output_shapes
:



*
use_locking(
А
save/Assign_7AssignVariable_15save/RestoreV2:7*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*&
_output_shapes
:

А
save/Assign_8AssignVariable_16save/RestoreV2:8*
use_locking(*
T0*
_class
loc:@Variable_16*
validate_shape(*&
_output_shapes
:




А
save/Assign_9AssignVariable_17save/RestoreV2:9*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(
В
save/Assign_10AssignVariable_18save/RestoreV2:10*&
_output_shapes
:



*
use_locking(*
T0*
_class
loc:@Variable_18*
validate_shape(
В
save/Assign_11AssignVariable_19save/RestoreV2:11*
T0*
_class
loc:@Variable_19*
validate_shape(*&
_output_shapes
:
*
use_locking(
А
save/Assign_12Assign
Variable_2save/RestoreV2:12*
T0*
_class
loc:@Variable_2*
validate_shape(*&
_output_shapes
:



*
use_locking(
В
save/Assign_13AssignVariable_20save/RestoreV2:13*
use_locking(*
T0*
_class
loc:@Variable_20*
validate_shape(*&
_output_shapes
:




В
save/Assign_14AssignVariable_21save/RestoreV2:14*
_class
loc:@Variable_21*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0
В
save/Assign_15AssignVariable_22save/RestoreV2:15*
_class
loc:@Variable_22*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
В
save/Assign_16AssignVariable_23save/RestoreV2:16*
_class
loc:@Variable_23*
validate_shape(*&
_output_shapes
:
*
use_locking(*
T0
В
save/Assign_17AssignVariable_24save/RestoreV2:17*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(*&
_output_shapes
:




В
save/Assign_18AssignVariable_25save/RestoreV2:18*&
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(
Ћ
save/Assign_19AssignVariable_26save/RestoreV2:19*
_output_shapes
:	ъ	*
use_locking(*
T0*
_class
loc:@Variable_26*
validate_shape(
Њ
save/Assign_20AssignVariable_27save/RestoreV2:20*
_output_shapes

:	*
use_locking(*
T0*
_class
loc:@Variable_27*
validate_shape(
А
save/Assign_21Assign
Variable_3save/RestoreV2:21*
T0*
_class
loc:@Variable_3*
validate_shape(*&
_output_shapes
:22
*
use_locking(
А
save/Assign_22Assign
Variable_4save/RestoreV2:22*
_class
loc:@Variable_4*
validate_shape(*&
_output_shapes
:



*
use_locking(*
T0
А
save/Assign_23Assign
Variable_5save/RestoreV2:23*
use_locking(*
T0*
_class
loc:@Variable_5*
validate_shape(*&
_output_shapes
:22

А
save/Assign_24Assign
Variable_6save/RestoreV2:24*
use_locking(*
T0*
_class
loc:@Variable_6*
validate_shape(*&
_output_shapes
:




А
save/Assign_25Assign
Variable_7save/RestoreV2:25*
validate_shape(*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_7
А
save/Assign_26Assign
Variable_8save/RestoreV2:26*
use_locking(*
T0*
_class
loc:@Variable_8*
validate_shape(*&
_output_shapes
:




А
save/Assign_27Assign
Variable_9save/RestoreV2:27*
validate_shape(*&
_output_shapes
:22
*
use_locking(*
T0*
_class
loc:@Variable_9
ш
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27""
train_op

Adam"ќA
	variablesюAыA
:

Variable:0Variable/AssignVariable/read:02
Kernel_0:0
E
Variable_1:0Variable_1/AssignVariable_1/read:02Kernel_Bias_0:0
@
Variable_2:0Variable_2/AssignVariable_2/read:02
Kernel_1:0
E
Variable_3:0Variable_3/AssignVariable_3/read:02Kernel_Bias_1:0
@
Variable_4:0Variable_4/AssignVariable_4/read:02
Kernel_2:0
E
Variable_5:0Variable_5/AssignVariable_5/read:02Kernel_Bias_2:0
@
Variable_6:0Variable_6/AssignVariable_6/read:02
Kernel_3:0
E
Variable_7:0Variable_7/AssignVariable_7/read:02Kernel_Bias_3:0
@
Variable_8:0Variable_8/AssignVariable_8/read:02
Kernel_4:0
E
Variable_9:0Variable_9/AssignVariable_9/read:02Kernel_Bias_4:0
C
Variable_10:0Variable_10/AssignVariable_10/read:02
Kernel_5:0
H
Variable_11:0Variable_11/AssignVariable_11/read:02Kernel_Bias_5:0
C
Variable_12:0Variable_12/AssignVariable_12/read:02
Kernel_6:0
H
Variable_13:0Variable_13/AssignVariable_13/read:02Kernel_Bias_6:0
C
Variable_14:0Variable_14/AssignVariable_14/read:02
Kernel_7:0
H
Variable_15:0Variable_15/AssignVariable_15/read:02Kernel_Bias_7:0
C
Variable_16:0Variable_16/AssignVariable_16/read:02
Kernel_8:0
H
Variable_17:0Variable_17/AssignVariable_17/read:02Kernel_Bias_8:0
C
Variable_18:0Variable_18/AssignVariable_18/read:02
Kernel_9:0
H
Variable_19:0Variable_19/AssignVariable_19/read:02Kernel_Bias_9:0
D
Variable_20:0Variable_20/AssignVariable_20/read:02Kernel_10:0
I
Variable_21:0Variable_21/AssignVariable_21/read:02Kernel_Bias_10:0
D
Variable_22:0Variable_22/AssignVariable_22/read:02Kernel_11:0
I
Variable_23:0Variable_23/AssignVariable_23/read:02Kernel_Bias_11:0
D
Variable_24:0Variable_24/AssignVariable_24/read:02Kernel_12:0
I
Variable_25:0Variable_25/AssignVariable_25/read:02Kernel_Bias_12:0
D
Variable_26:0Variable_26/AssignVariable_26/read:02FC_Weight:0
B
Variable_27:0Variable_27/AssignVariable_27/read:02	FC_Bias:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
`
Variable/Adam:0Variable/Adam/AssignVariable/Adam/read:02!Variable/Adam/Initializer/zeros:0
h
Variable/Adam_1:0Variable/Adam_1/AssignVariable/Adam_1/read:02#Variable/Adam_1/Initializer/zeros:0
h
Variable_1/Adam:0Variable_1/Adam/AssignVariable_1/Adam/read:02#Variable_1/Adam/Initializer/zeros:0
p
Variable_1/Adam_1:0Variable_1/Adam_1/AssignVariable_1/Adam_1/read:02%Variable_1/Adam_1/Initializer/zeros:0
h
Variable_2/Adam:0Variable_2/Adam/AssignVariable_2/Adam/read:02#Variable_2/Adam/Initializer/zeros:0
p
Variable_2/Adam_1:0Variable_2/Adam_1/AssignVariable_2/Adam_1/read:02%Variable_2/Adam_1/Initializer/zeros:0
h
Variable_3/Adam:0Variable_3/Adam/AssignVariable_3/Adam/read:02#Variable_3/Adam/Initializer/zeros:0
p
Variable_3/Adam_1:0Variable_3/Adam_1/AssignVariable_3/Adam_1/read:02%Variable_3/Adam_1/Initializer/zeros:0
h
Variable_4/Adam:0Variable_4/Adam/AssignVariable_4/Adam/read:02#Variable_4/Adam/Initializer/zeros:0
p
Variable_4/Adam_1:0Variable_4/Adam_1/AssignVariable_4/Adam_1/read:02%Variable_4/Adam_1/Initializer/zeros:0
h
Variable_5/Adam:0Variable_5/Adam/AssignVariable_5/Adam/read:02#Variable_5/Adam/Initializer/zeros:0
p
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0
h
Variable_6/Adam:0Variable_6/Adam/AssignVariable_6/Adam/read:02#Variable_6/Adam/Initializer/zeros:0
p
Variable_6/Adam_1:0Variable_6/Adam_1/AssignVariable_6/Adam_1/read:02%Variable_6/Adam_1/Initializer/zeros:0
h
Variable_7/Adam:0Variable_7/Adam/AssignVariable_7/Adam/read:02#Variable_7/Adam/Initializer/zeros:0
p
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0
h
Variable_8/Adam:0Variable_8/Adam/AssignVariable_8/Adam/read:02#Variable_8/Adam/Initializer/zeros:0
p
Variable_8/Adam_1:0Variable_8/Adam_1/AssignVariable_8/Adam_1/read:02%Variable_8/Adam_1/Initializer/zeros:0
h
Variable_9/Adam:0Variable_9/Adam/AssignVariable_9/Adam/read:02#Variable_9/Adam/Initializer/zeros:0
p
Variable_9/Adam_1:0Variable_9/Adam_1/AssignVariable_9/Adam_1/read:02%Variable_9/Adam_1/Initializer/zeros:0
l
Variable_10/Adam:0Variable_10/Adam/AssignVariable_10/Adam/read:02$Variable_10/Adam/Initializer/zeros:0
t
Variable_10/Adam_1:0Variable_10/Adam_1/AssignVariable_10/Adam_1/read:02&Variable_10/Adam_1/Initializer/zeros:0
l
Variable_11/Adam:0Variable_11/Adam/AssignVariable_11/Adam/read:02$Variable_11/Adam/Initializer/zeros:0
t
Variable_11/Adam_1:0Variable_11/Adam_1/AssignVariable_11/Adam_1/read:02&Variable_11/Adam_1/Initializer/zeros:0
l
Variable_12/Adam:0Variable_12/Adam/AssignVariable_12/Adam/read:02$Variable_12/Adam/Initializer/zeros:0
t
Variable_12/Adam_1:0Variable_12/Adam_1/AssignVariable_12/Adam_1/read:02&Variable_12/Adam_1/Initializer/zeros:0
l
Variable_13/Adam:0Variable_13/Adam/AssignVariable_13/Adam/read:02$Variable_13/Adam/Initializer/zeros:0
t
Variable_13/Adam_1:0Variable_13/Adam_1/AssignVariable_13/Adam_1/read:02&Variable_13/Adam_1/Initializer/zeros:0
l
Variable_14/Adam:0Variable_14/Adam/AssignVariable_14/Adam/read:02$Variable_14/Adam/Initializer/zeros:0
t
Variable_14/Adam_1:0Variable_14/Adam_1/AssignVariable_14/Adam_1/read:02&Variable_14/Adam_1/Initializer/zeros:0
l
Variable_15/Adam:0Variable_15/Adam/AssignVariable_15/Adam/read:02$Variable_15/Adam/Initializer/zeros:0
t
Variable_15/Adam_1:0Variable_15/Adam_1/AssignVariable_15/Adam_1/read:02&Variable_15/Adam_1/Initializer/zeros:0
l
Variable_16/Adam:0Variable_16/Adam/AssignVariable_16/Adam/read:02$Variable_16/Adam/Initializer/zeros:0
t
Variable_16/Adam_1:0Variable_16/Adam_1/AssignVariable_16/Adam_1/read:02&Variable_16/Adam_1/Initializer/zeros:0
l
Variable_17/Adam:0Variable_17/Adam/AssignVariable_17/Adam/read:02$Variable_17/Adam/Initializer/zeros:0
t
Variable_17/Adam_1:0Variable_17/Adam_1/AssignVariable_17/Adam_1/read:02&Variable_17/Adam_1/Initializer/zeros:0
l
Variable_18/Adam:0Variable_18/Adam/AssignVariable_18/Adam/read:02$Variable_18/Adam/Initializer/zeros:0
t
Variable_18/Adam_1:0Variable_18/Adam_1/AssignVariable_18/Adam_1/read:02&Variable_18/Adam_1/Initializer/zeros:0
l
Variable_19/Adam:0Variable_19/Adam/AssignVariable_19/Adam/read:02$Variable_19/Adam/Initializer/zeros:0
t
Variable_19/Adam_1:0Variable_19/Adam_1/AssignVariable_19/Adam_1/read:02&Variable_19/Adam_1/Initializer/zeros:0
l
Variable_20/Adam:0Variable_20/Adam/AssignVariable_20/Adam/read:02$Variable_20/Adam/Initializer/zeros:0
t
Variable_20/Adam_1:0Variable_20/Adam_1/AssignVariable_20/Adam_1/read:02&Variable_20/Adam_1/Initializer/zeros:0
l
Variable_21/Adam:0Variable_21/Adam/AssignVariable_21/Adam/read:02$Variable_21/Adam/Initializer/zeros:0
t
Variable_21/Adam_1:0Variable_21/Adam_1/AssignVariable_21/Adam_1/read:02&Variable_21/Adam_1/Initializer/zeros:0
l
Variable_22/Adam:0Variable_22/Adam/AssignVariable_22/Adam/read:02$Variable_22/Adam/Initializer/zeros:0
t
Variable_22/Adam_1:0Variable_22/Adam_1/AssignVariable_22/Adam_1/read:02&Variable_22/Adam_1/Initializer/zeros:0
l
Variable_23/Adam:0Variable_23/Adam/AssignVariable_23/Adam/read:02$Variable_23/Adam/Initializer/zeros:0
t
Variable_23/Adam_1:0Variable_23/Adam_1/AssignVariable_23/Adam_1/read:02&Variable_23/Adam_1/Initializer/zeros:0
l
Variable_24/Adam:0Variable_24/Adam/AssignVariable_24/Adam/read:02$Variable_24/Adam/Initializer/zeros:0
t
Variable_24/Adam_1:0Variable_24/Adam_1/AssignVariable_24/Adam_1/read:02&Variable_24/Adam_1/Initializer/zeros:0
l
Variable_25/Adam:0Variable_25/Adam/AssignVariable_25/Adam/read:02$Variable_25/Adam/Initializer/zeros:0
t
Variable_25/Adam_1:0Variable_25/Adam_1/AssignVariable_25/Adam_1/read:02&Variable_25/Adam_1/Initializer/zeros:0
l
Variable_26/Adam:0Variable_26/Adam/AssignVariable_26/Adam/read:02$Variable_26/Adam/Initializer/zeros:0
t
Variable_26/Adam_1:0Variable_26/Adam_1/AssignVariable_26/Adam_1/read:02&Variable_26/Adam_1/Initializer/zeros:0
l
Variable_27/Adam:0Variable_27/Adam/AssignVariable_27/Adam/read:02$Variable_27/Adam/Initializer/zeros:0
t
Variable_27/Adam_1:0Variable_27/Adam_1/AssignVariable_27/Adam_1/read:02&Variable_27/Adam_1/Initializer/zeros:0"Ъ
trainable_variablesВЏ
:

Variable:0Variable/AssignVariable/read:02
Kernel_0:0
E
Variable_1:0Variable_1/AssignVariable_1/read:02Kernel_Bias_0:0
@
Variable_2:0Variable_2/AssignVariable_2/read:02
Kernel_1:0
E
Variable_3:0Variable_3/AssignVariable_3/read:02Kernel_Bias_1:0
@
Variable_4:0Variable_4/AssignVariable_4/read:02
Kernel_2:0
E
Variable_5:0Variable_5/AssignVariable_5/read:02Kernel_Bias_2:0
@
Variable_6:0Variable_6/AssignVariable_6/read:02
Kernel_3:0
E
Variable_7:0Variable_7/AssignVariable_7/read:02Kernel_Bias_3:0
@
Variable_8:0Variable_8/AssignVariable_8/read:02
Kernel_4:0
E
Variable_9:0Variable_9/AssignVariable_9/read:02Kernel_Bias_4:0
C
Variable_10:0Variable_10/AssignVariable_10/read:02
Kernel_5:0
H
Variable_11:0Variable_11/AssignVariable_11/read:02Kernel_Bias_5:0
C
Variable_12:0Variable_12/AssignVariable_12/read:02
Kernel_6:0
H
Variable_13:0Variable_13/AssignVariable_13/read:02Kernel_Bias_6:0
C
Variable_14:0Variable_14/AssignVariable_14/read:02
Kernel_7:0
H
Variable_15:0Variable_15/AssignVariable_15/read:02Kernel_Bias_7:0
C
Variable_16:0Variable_16/AssignVariable_16/read:02
Kernel_8:0
H
Variable_17:0Variable_17/AssignVariable_17/read:02Kernel_Bias_8:0
C
Variable_18:0Variable_18/AssignVariable_18/read:02
Kernel_9:0
H
Variable_19:0Variable_19/AssignVariable_19/read:02Kernel_Bias_9:0
D
Variable_20:0Variable_20/AssignVariable_20/read:02Kernel_10:0
I
Variable_21:0Variable_21/AssignVariable_21/read:02Kernel_Bias_10:0
D
Variable_22:0Variable_22/AssignVariable_22/read:02Kernel_11:0
I
Variable_23:0Variable_23/AssignVariable_23/read:02Kernel_Bias_11:0
D
Variable_24:0Variable_24/AssignVariable_24/read:02Kernel_12:0
I
Variable_25:0Variable_25/AssignVariable_25/read:02Kernel_Bias_12:0
D
Variable_26:0Variable_26/AssignVariable_26/read:02FC_Weight:0
B
Variable_27:0Variable_27/AssignVariable_27/read:02	FC_Bias:0)8CЉ