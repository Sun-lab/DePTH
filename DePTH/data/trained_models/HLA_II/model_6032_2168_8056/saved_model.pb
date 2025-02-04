̶
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
�
false_negativesVarHandleOp*
_output_shapes
: * 

debug_namefalse_negatives/*
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
�
false_positivesVarHandleOp*
_output_shapes
: * 

debug_namefalse_positives/*
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
�
true_negativesVarHandleOp*
_output_shapes
: *

debug_nametrue_negatives/*
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
�
true_positivesVarHandleOp*
_output_shapes
: *

debug_nametrue_positives/*
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/output/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/v/output/bias/*
dtype0*
shape:*#
shared_nameAdam/v/output/bias
u
&Adam/v/output/bias/Read/ReadVariableOpReadVariableOpAdam/v/output/bias*
_output_shapes
:*
dtype0
�
Adam/m/output/biasVarHandleOp*
_output_shapes
: *#

debug_nameAdam/m/output/bias/*
dtype0*
shape:*#
shared_nameAdam/m/output/bias
u
&Adam/m/output/bias/Read/ReadVariableOpReadVariableOpAdam/m/output/bias*
_output_shapes
:*
dtype0
�
Adam/v/output/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/output/kernel/*
dtype0*
shape
:*%
shared_nameAdam/v/output/kernel
}
(Adam/v/output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/output/kernel*
_output_shapes

:*
dtype0
�
Adam/m/output/kernelVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/output/kernel/*
dtype0*
shape
:*%
shared_nameAdam/m/output/kernel
}
(Adam/m/output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/output/kernel*
_output_shapes

:*
dtype0
�
Adam/v/pair_dense2/biasVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/pair_dense2/bias/*
dtype0*
shape:*(
shared_nameAdam/v/pair_dense2/bias

+Adam/v/pair_dense2/bias/Read/ReadVariableOpReadVariableOpAdam/v/pair_dense2/bias*
_output_shapes
:*
dtype0
�
Adam/m/pair_dense2/biasVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/pair_dense2/bias/*
dtype0*
shape:*(
shared_nameAdam/m/pair_dense2/bias

+Adam/m/pair_dense2/bias/Read/ReadVariableOpReadVariableOpAdam/m/pair_dense2/bias*
_output_shapes
:*
dtype0
�
Adam/v/pair_dense2/kernelVarHandleOp*
_output_shapes
: **

debug_nameAdam/v/pair_dense2/kernel/*
dtype0*
shape
:@**
shared_nameAdam/v/pair_dense2/kernel
�
-Adam/v/pair_dense2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/pair_dense2/kernel*
_output_shapes

:@*
dtype0
�
Adam/m/pair_dense2/kernelVarHandleOp*
_output_shapes
: **

debug_nameAdam/m/pair_dense2/kernel/*
dtype0*
shape
:@**
shared_nameAdam/m/pair_dense2/kernel
�
-Adam/m/pair_dense2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/pair_dense2/kernel*
_output_shapes

:@*
dtype0
�
Adam/v/pair_dense1/biasVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/pair_dense1/bias/*
dtype0*
shape:@*(
shared_nameAdam/v/pair_dense1/bias

+Adam/v/pair_dense1/bias/Read/ReadVariableOpReadVariableOpAdam/v/pair_dense1/bias*
_output_shapes
:@*
dtype0
�
Adam/m/pair_dense1/biasVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/pair_dense1/bias/*
dtype0*
shape:@*(
shared_nameAdam/m/pair_dense1/bias

+Adam/m/pair_dense1/bias/Read/ReadVariableOpReadVariableOpAdam/m/pair_dense1/bias*
_output_shapes
:@*
dtype0
�
Adam/v/pair_dense1/kernelVarHandleOp*
_output_shapes
: **

debug_nameAdam/v/pair_dense1/kernel/*
dtype0*
shape:	�@**
shared_nameAdam/v/pair_dense1/kernel
�
-Adam/v/pair_dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/pair_dense1/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/pair_dense1/kernelVarHandleOp*
_output_shapes
: **

debug_nameAdam/m/pair_dense1/kernel/*
dtype0*
shape:	�@**
shared_nameAdam/m/pair_dense1/kernel
�
-Adam/m/pair_dense1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/pair_dense1/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/TCR_dense/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/TCR_dense/bias/*
dtype0*
shape:@*&
shared_nameAdam/v/TCR_dense/bias
{
)Adam/v/TCR_dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/TCR_dense/bias*
_output_shapes
:@*
dtype0
�
Adam/m/TCR_dense/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/TCR_dense/bias/*
dtype0*
shape:@*&
shared_nameAdam/m/TCR_dense/bias
{
)Adam/m/TCR_dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/TCR_dense/bias*
_output_shapes
:@*
dtype0
�
Adam/v/TCR_dense/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/TCR_dense/kernel/*
dtype0*
shape:	�
@*(
shared_nameAdam/v/TCR_dense/kernel
�
+Adam/v/TCR_dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/TCR_dense/kernel*
_output_shapes
:	�
@*
dtype0
�
Adam/m/TCR_dense/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/TCR_dense/kernel/*
dtype0*
shape:	�
@*(
shared_nameAdam/m/TCR_dense/kernel
�
+Adam/m/TCR_dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/TCR_dense/kernel*
_output_shapes
:	�
@*
dtype0
�
Adam/v/HLA_dense/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/HLA_dense/bias/*
dtype0*
shape:@*&
shared_nameAdam/v/HLA_dense/bias
{
)Adam/v/HLA_dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/HLA_dense/bias*
_output_shapes
:@*
dtype0
�
Adam/m/HLA_dense/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/HLA_dense/bias/*
dtype0*
shape:@*&
shared_nameAdam/m/HLA_dense/bias
{
)Adam/m/HLA_dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/HLA_dense/bias*
_output_shapes
:@*
dtype0
�
Adam/v/HLA_dense/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/HLA_dense/kernel/*
dtype0*
shape:	�@*(
shared_nameAdam/v/HLA_dense/kernel
�
+Adam/v/HLA_dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/HLA_dense/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/HLA_dense/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/HLA_dense/kernel/*
dtype0*
shape:	�@*(
shared_nameAdam/m/HLA_dense/kernel
�
+Adam/m/HLA_dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/HLA_dense/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/CDR3_conv/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/CDR3_conv/bias/*
dtype0*
shape:*&
shared_nameAdam/v/CDR3_conv/bias
{
)Adam/v/CDR3_conv/bias/Read/ReadVariableOpReadVariableOpAdam/v/CDR3_conv/bias*
_output_shapes
:*
dtype0
�
Adam/m/CDR3_conv/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/CDR3_conv/bias/*
dtype0*
shape:*&
shared_nameAdam/m/CDR3_conv/bias
{
)Adam/m/CDR3_conv/bias/Read/ReadVariableOpReadVariableOpAdam/m/CDR3_conv/bias*
_output_shapes
:*
dtype0
�
Adam/v/CDR3_conv/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/CDR3_conv/kernel/*
dtype0*
shape:*(
shared_nameAdam/v/CDR3_conv/kernel
�
+Adam/v/CDR3_conv/kernel/Read/ReadVariableOpReadVariableOpAdam/v/CDR3_conv/kernel*"
_output_shapes
:*
dtype0
�
Adam/m/CDR3_conv/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/CDR3_conv/kernel/*
dtype0*
shape:*(
shared_nameAdam/m/CDR3_conv/kernel
�
+Adam/m/CDR3_conv/kernel/Read/ReadVariableOpReadVariableOpAdam/m/CDR3_conv/kernel*"
_output_shapes
:*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
output/biasVarHandleOp*
_output_shapes
: *

debug_nameoutput/bias/*
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
�
output/kernelVarHandleOp*
_output_shapes
: *

debug_nameoutput/kernel/*
dtype0*
shape
:*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:*
dtype0
�
pair_dense2/biasVarHandleOp*
_output_shapes
: *!

debug_namepair_dense2/bias/*
dtype0*
shape:*!
shared_namepair_dense2/bias
q
$pair_dense2/bias/Read/ReadVariableOpReadVariableOppair_dense2/bias*
_output_shapes
:*
dtype0
�
pair_dense2/kernelVarHandleOp*
_output_shapes
: *#

debug_namepair_dense2/kernel/*
dtype0*
shape
:@*#
shared_namepair_dense2/kernel
y
&pair_dense2/kernel/Read/ReadVariableOpReadVariableOppair_dense2/kernel*
_output_shapes

:@*
dtype0
�
pair_dense1/biasVarHandleOp*
_output_shapes
: *!

debug_namepair_dense1/bias/*
dtype0*
shape:@*!
shared_namepair_dense1/bias
q
$pair_dense1/bias/Read/ReadVariableOpReadVariableOppair_dense1/bias*
_output_shapes
:@*
dtype0
�
pair_dense1/kernelVarHandleOp*
_output_shapes
: *#

debug_namepair_dense1/kernel/*
dtype0*
shape:	�@*#
shared_namepair_dense1/kernel
z
&pair_dense1/kernel/Read/ReadVariableOpReadVariableOppair_dense1/kernel*
_output_shapes
:	�@*
dtype0
�
TCR_dense/biasVarHandleOp*
_output_shapes
: *

debug_nameTCR_dense/bias/*
dtype0*
shape:@*
shared_nameTCR_dense/bias
m
"TCR_dense/bias/Read/ReadVariableOpReadVariableOpTCR_dense/bias*
_output_shapes
:@*
dtype0
�
TCR_dense/kernelVarHandleOp*
_output_shapes
: *!

debug_nameTCR_dense/kernel/*
dtype0*
shape:	�
@*!
shared_nameTCR_dense/kernel
v
$TCR_dense/kernel/Read/ReadVariableOpReadVariableOpTCR_dense/kernel*
_output_shapes
:	�
@*
dtype0
�
HLA_dense/biasVarHandleOp*
_output_shapes
: *

debug_nameHLA_dense/bias/*
dtype0*
shape:@*
shared_nameHLA_dense/bias
m
"HLA_dense/bias/Read/ReadVariableOpReadVariableOpHLA_dense/bias*
_output_shapes
:@*
dtype0
�
HLA_dense/kernelVarHandleOp*
_output_shapes
: *!

debug_nameHLA_dense/kernel/*
dtype0*
shape:	�@*!
shared_nameHLA_dense/kernel
v
$HLA_dense/kernel/Read/ReadVariableOpReadVariableOpHLA_dense/kernel*
_output_shapes
:	�@*
dtype0
�
CDR3_conv/biasVarHandleOp*
_output_shapes
: *

debug_nameCDR3_conv/bias/*
dtype0*
shape:*
shared_nameCDR3_conv/bias
m
"CDR3_conv/bias/Read/ReadVariableOpReadVariableOpCDR3_conv/bias*
_output_shapes
:*
dtype0
�
CDR3_conv/kernelVarHandleOp*
_output_shapes
: *!

debug_nameCDR3_conv/kernel/*
dtype0*
shape:*!
shared_nameCDR3_conv/kernel
y
$CDR3_conv/kernel/Read/ReadVariableOpReadVariableOpCDR3_conv/kernel*"
_output_shapes
:*
dtype0

serving_default_CDR1Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������

serving_default_CDR2Placeholder*+
_output_shapes
:���������
*
dtype0* 
shape:���������

�
serving_default_CDR2.5Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������

serving_default_CDR3Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
{
serving_default_CDR3_lenPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
~
serving_default_HLAPlaceholder*+
_output_shapes
:���������-*
dtype0* 
shape:���������-
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_CDR1serving_default_CDR2serving_default_CDR2.5serving_default_CDR3serving_default_CDR3_lenserving_default_HLACDR3_conv/kernelCDR3_conv/biasHLA_dense/kernelHLA_dense/biasTCR_dense/kernelTCR_dense/biaspair_dense1/kernelpair_dense1/biaspair_dense2/kernelpair_dense2/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_356169

NoOpNoOp
�{
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�{
value�{B�{ B�z
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-1
layer-16
layer_with_weights-2
layer-17
layer-18
layer_with_weights-3
layer-19
layer-20
layer_with_weights-4
layer-21
layer_with_weights-5
layer-22
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
 
signatures*
* 
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op*
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses* 
* 
* 
* 
* 
* 
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses* 
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 
�
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses* 
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias*
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias*
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias*
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
^
'0
(1
f2
g3
n4
o5
|6
}7
�8
�9
�10
�11*
^
'0
(1
f2
g3
n4
o5
|6
}7
�8
�9
�10
�11*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

'0
(1*

'0
(1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUECDR3_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUECDR3_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

f0
g1*

f0
g1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEHLA_dense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEHLA_dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

n0
o1*

n0
o1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUETCR_dense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUETCR_dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

|0
}1*

|0
}1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEpair_dense1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEpair_dense1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEpair_dense2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEpair_dense2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEoutput/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEoutput/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22*

�0
�1*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
z
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives*
b\
VARIABLE_VALUEAdam/m/CDR3_conv/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/CDR3_conv/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/CDR3_conv/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/CDR3_conv/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/HLA_dense/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/HLA_dense/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/HLA_dense/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/HLA_dense/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/TCR_dense/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/TCR_dense/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/TCR_dense/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/TCR_dense/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/pair_dense1/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/pair_dense1/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/pair_dense1/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/pair_dense1/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/pair_dense2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/pair_dense2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/pair_dense2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/pair_dense2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/output/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/output/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/output/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/output/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameCDR3_conv/kernelCDR3_conv/biasHLA_dense/kernelHLA_dense/biasTCR_dense/kernelTCR_dense/biaspair_dense1/kernelpair_dense1/biaspair_dense2/kernelpair_dense2/biasoutput/kerneloutput/bias	iterationlearning_rateAdam/m/CDR3_conv/kernelAdam/v/CDR3_conv/kernelAdam/m/CDR3_conv/biasAdam/v/CDR3_conv/biasAdam/m/HLA_dense/kernelAdam/v/HLA_dense/kernelAdam/m/HLA_dense/biasAdam/v/HLA_dense/biasAdam/m/TCR_dense/kernelAdam/v/TCR_dense/kernelAdam/m/TCR_dense/biasAdam/v/TCR_dense/biasAdam/m/pair_dense1/kernelAdam/v/pair_dense1/kernelAdam/m/pair_dense1/biasAdam/v/pair_dense1/biasAdam/m/pair_dense2/kernelAdam/v/pair_dense2/kernelAdam/m/pair_dense2/biasAdam/v/pair_dense2/biasAdam/m/output/kernelAdam/v/output/kernelAdam/m/output/biasAdam/v/output/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativesConst*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_356766
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameCDR3_conv/kernelCDR3_conv/biasHLA_dense/kernelHLA_dense/biasTCR_dense/kernelTCR_dense/biaspair_dense1/kernelpair_dense1/biaspair_dense2/kernelpair_dense2/biasoutput/kerneloutput/bias	iterationlearning_rateAdam/m/CDR3_conv/kernelAdam/v/CDR3_conv/kernelAdam/m/CDR3_conv/biasAdam/v/CDR3_conv/biasAdam/m/HLA_dense/kernelAdam/v/HLA_dense/kernelAdam/m/HLA_dense/biasAdam/v/HLA_dense/biasAdam/m/TCR_dense/kernelAdam/v/TCR_dense/kernelAdam/m/TCR_dense/biasAdam/v/TCR_dense/biasAdam/m/pair_dense1/kernelAdam/v/pair_dense1/kernelAdam/m/pair_dense1/biasAdam/v/pair_dense1/biasAdam/m/pair_dense2/kernelAdam/v/pair_dense2/kernelAdam/m/pair_dense2/biasAdam/v/pair_dense2/biasAdam/m/output/kernelAdam/v/output/kernelAdam/m/output/biasAdam/v/output/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_356907Ĳ

�
I
-__inference_CDR3_flatten_layer_call_fn_356212

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR3_flatten_layer_call_and_return_conditional_losses_355706a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
d
H__inference_CDR2_reshape_layer_call_and_return_conditional_losses_356282

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
*__inference_HLA_dense_layer_call_fn_356344

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_HLA_dense_layer_call_and_return_conditional_losses_355802o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name356340:&"
 
_user_specified_name356338:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_356435

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
I
-__inference_CDR2_reshape_layer_call_fn_356270

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR2_reshape_layer_call_and_return_conditional_losses_355753a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
g
K__inference_CDR3_maxpooling_layer_call_and_return_conditional_losses_355666

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+����������������������������
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�

�
E__inference_HLA_dense_layer_call_and_return_conditional_losses_356355

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_356017
hla
cdr3
cdr3_len
cdr1
cdr2

cdr2_5
unknown:
	unknown_0:
	unknown_1:	�@
	unknown_2:@
	unknown_3:	�
@
	unknown_4:@
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallhlacdr3cdr3_lencdr1cdr2cdr2_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_355949o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name356013:&"
 
_user_specified_name356011:&"
 
_user_specified_name356009:&"
 
_user_specified_name356007:&"
 
_user_specified_name356005:&"
 
_user_specified_name356003:&"
 
_user_specified_name356001:&
"
 
_user_specified_name355999:&	"
 
_user_specified_name355997:&"
 
_user_specified_name355995:&"
 
_user_specified_name355993:&"
 
_user_specified_name355991:SO
+
_output_shapes
:���������
 
_user_specified_nameCDR2.5:QM
+
_output_shapes
:���������


_user_specified_nameCDR2:QM
+
_output_shapes
:���������

_user_specified_nameCDR1:QM
'
_output_shapes
:���������
"
_user_specified_name
CDR3_len:QM
+
_output_shapes
:���������

_user_specified_nameCDR3:P L
+
_output_shapes
:���������-

_user_specified_nameHLA
�
d
H__inference_CDR3_flatten_layer_call_and_return_conditional_losses_355706

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�H
�
A__inference_model_layer_call_and_return_conditional_losses_355949
hla
cdr3
cdr3_len
cdr1
cdr2

cdr2_5&
cdr3_conv_355902:
cdr3_conv_355904:#
hla_dense_355916:	�@
hla_dense_355918:@#
tcr_dense_355921:	�
@
tcr_dense_355923:@%
pair_dense1_355927:	�@ 
pair_dense1_355929:@$
pair_dense2_355938:@ 
pair_dense2_355940:
output_355943:
output_355945:
identity��!CDR3_conv/StatefulPartitionedCall�!HLA_dense/StatefulPartitionedCall�!TCR_dense/StatefulPartitionedCall�output/StatefulPartitionedCall�#pair_dense1/StatefulPartitionedCall�#pair_dense2/StatefulPartitionedCall�
!CDR3_conv/StatefulPartitionedCallStatefulPartitionedCallcdr3cdr3_conv_355902cdr3_conv_355904*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_CDR3_conv_layer_call_and_return_conditional_losses_355694�
CDR3_maxpooling/PartitionedCallPartitionedCall*CDR3_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_CDR3_maxpooling_layer_call_and_return_conditional_losses_355666�
CDR3_flatten/PartitionedCallPartitionedCall(CDR3_maxpooling/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR3_flatten_layer_call_and_return_conditional_losses_355706�
CDR3_reshape/PartitionedCallPartitionedCallcdr3*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR3_reshape_layer_call_and_return_conditional_losses_355719�
concatenate/PartitionedCallPartitionedCall%CDR3_flatten/PartitionedCall:output:0%CDR3_reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_355727�
CDR1_reshape/PartitionedCallPartitionedCallcdr1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR1_reshape_layer_call_and_return_conditional_losses_355740�
CDR2_reshape/PartitionedCallPartitionedCallcdr2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR2_reshape_layer_call_and_return_conditional_losses_355753�
CDR2.5_reshape/PartitionedCallPartitionedCallcdr2_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������~* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_CDR2.5_reshape_layer_call_and_return_conditional_losses_355766�
concatenate_1/PartitionedCallPartitionedCallcdr3_len$concatenate/PartitionedCall:output:0%CDR1_reshape/PartitionedCall:output:0%CDR2_reshape/PartitionedCall:output:0'CDR2.5_reshape/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_355777�
HLA_reshape/PartitionedCallPartitionedCallhla*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_HLA_reshape_layer_call_and_return_conditional_losses_355790�
!HLA_dense/StatefulPartitionedCallStatefulPartitionedCall$HLA_reshape/PartitionedCall:output:0hla_dense_355916hla_dense_355918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_HLA_dense_layer_call_and_return_conditional_losses_355802�
!TCR_dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0tcr_dense_355921tcr_dense_355923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_TCR_dense_layer_call_and_return_conditional_losses_355818�
concatenate_2/PartitionedCallPartitionedCall*HLA_dense/StatefulPartitionedCall:output:0*TCR_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_355830�
#pair_dense1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0pair_dense1_355927pair_dense1_355929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_pair_dense1_layer_call_and_return_conditional_losses_355842�
dropout/PartitionedCallPartitionedCall,pair_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_355936�
#pair_dense2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0pair_dense2_355938pair_dense2_355940*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_pair_dense2_layer_call_and_return_conditional_losses_355871�
output/StatefulPartitionedCallStatefulPartitionedCall,pair_dense2/StatefulPartitionedCall:output:0output_355943output_355945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_355887v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^CDR3_conv/StatefulPartitionedCall"^HLA_dense/StatefulPartitionedCall"^TCR_dense/StatefulPartitionedCall^output/StatefulPartitionedCall$^pair_dense1/StatefulPartitionedCall$^pair_dense2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2F
!CDR3_conv/StatefulPartitionedCall!CDR3_conv/StatefulPartitionedCall2F
!HLA_dense/StatefulPartitionedCall!HLA_dense/StatefulPartitionedCall2F
!TCR_dense/StatefulPartitionedCall!TCR_dense/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2J
#pair_dense1/StatefulPartitionedCall#pair_dense1/StatefulPartitionedCall2J
#pair_dense2/StatefulPartitionedCall#pair_dense2/StatefulPartitionedCall:&"
 
_user_specified_name355945:&"
 
_user_specified_name355943:&"
 
_user_specified_name355940:&"
 
_user_specified_name355938:&"
 
_user_specified_name355929:&"
 
_user_specified_name355927:&"
 
_user_specified_name355923:&
"
 
_user_specified_name355921:&	"
 
_user_specified_name355918:&"
 
_user_specified_name355916:&"
 
_user_specified_name355904:&"
 
_user_specified_name355902:SO
+
_output_shapes
:���������
 
_user_specified_nameCDR2.5:QM
+
_output_shapes
:���������


_user_specified_nameCDR2:QM
+
_output_shapes
:���������

_user_specified_nameCDR1:QM
'
_output_shapes
:���������
"
_user_specified_name
CDR3_len:QM
+
_output_shapes
:���������

_user_specified_nameCDR3:P L
+
_output_shapes
:���������-

_user_specified_nameHLA
�	
c
G__inference_HLA_reshape_layer_call_and_return_conditional_losses_356316

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-:S O
+
_output_shapes
:���������-
 
_user_specified_nameinputs
�	
d
H__inference_CDR2_reshape_layer_call_and_return_conditional_losses_355753

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
K
/__inference_CDR2.5_reshape_layer_call_fn_356287

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������~* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_CDR2.5_reshape_layer_call_and_return_conditional_losses_355766`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������~"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_355983
hla
cdr3
cdr3_len
cdr1
cdr2

cdr2_5
unknown:
	unknown_0:
	unknown_1:	�@
	unknown_2:@
	unknown_3:	�
@
	unknown_4:@
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallhlacdr3cdr3_lencdr1cdr2cdr2_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_355894o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name355979:&"
 
_user_specified_name355977:&"
 
_user_specified_name355975:&"
 
_user_specified_name355973:&"
 
_user_specified_name355971:&"
 
_user_specified_name355969:&"
 
_user_specified_name355967:&
"
 
_user_specified_name355965:&	"
 
_user_specified_name355963:&"
 
_user_specified_name355961:&"
 
_user_specified_name355959:&"
 
_user_specified_name355957:SO
+
_output_shapes
:���������
 
_user_specified_nameCDR2.5:QM
+
_output_shapes
:���������


_user_specified_nameCDR2:QM
+
_output_shapes
:���������

_user_specified_nameCDR1:QM
'
_output_shapes
:���������
"
_user_specified_name
CDR3_len:QM
+
_output_shapes
:���������

_user_specified_nameCDR3:P L
+
_output_shapes
:���������-

_user_specified_nameHLA
�
I
-__inference_CDR1_reshape_layer_call_fn_356253

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR1_reshape_layer_call_and_return_conditional_losses_355740a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_pair_dense1_layer_call_and_return_conditional_losses_355842

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_TCR_dense_layer_call_and_return_conditional_losses_356375

inputs1
matmul_readvariableop_resource:	�
@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�
s
G__inference_concatenate_layer_call_and_return_conditional_losses_356248
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs_0
��
�(
__inference__traced_save_356766
file_prefix=
'read_disablecopyonread_cdr3_conv_kernel:5
'read_1_disablecopyonread_cdr3_conv_bias:<
)read_2_disablecopyonread_hla_dense_kernel:	�@5
'read_3_disablecopyonread_hla_dense_bias:@<
)read_4_disablecopyonread_tcr_dense_kernel:	�
@5
'read_5_disablecopyonread_tcr_dense_bias:@>
+read_6_disablecopyonread_pair_dense1_kernel:	�@7
)read_7_disablecopyonread_pair_dense1_bias:@=
+read_8_disablecopyonread_pair_dense2_kernel:@7
)read_9_disablecopyonread_pair_dense2_bias:9
'read_10_disablecopyonread_output_kernel:3
%read_11_disablecopyonread_output_bias:-
#read_12_disablecopyonread_iteration:	 1
'read_13_disablecopyonread_learning_rate: G
1read_14_disablecopyonread_adam_m_cdr3_conv_kernel:G
1read_15_disablecopyonread_adam_v_cdr3_conv_kernel:=
/read_16_disablecopyonread_adam_m_cdr3_conv_bias:=
/read_17_disablecopyonread_adam_v_cdr3_conv_bias:D
1read_18_disablecopyonread_adam_m_hla_dense_kernel:	�@D
1read_19_disablecopyonread_adam_v_hla_dense_kernel:	�@=
/read_20_disablecopyonread_adam_m_hla_dense_bias:@=
/read_21_disablecopyonread_adam_v_hla_dense_bias:@D
1read_22_disablecopyonread_adam_m_tcr_dense_kernel:	�
@D
1read_23_disablecopyonread_adam_v_tcr_dense_kernel:	�
@=
/read_24_disablecopyonread_adam_m_tcr_dense_bias:@=
/read_25_disablecopyonread_adam_v_tcr_dense_bias:@F
3read_26_disablecopyonread_adam_m_pair_dense1_kernel:	�@F
3read_27_disablecopyonread_adam_v_pair_dense1_kernel:	�@?
1read_28_disablecopyonread_adam_m_pair_dense1_bias:@?
1read_29_disablecopyonread_adam_v_pair_dense1_bias:@E
3read_30_disablecopyonread_adam_m_pair_dense2_kernel:@E
3read_31_disablecopyonread_adam_v_pair_dense2_kernel:@?
1read_32_disablecopyonread_adam_m_pair_dense2_bias:?
1read_33_disablecopyonread_adam_v_pair_dense2_bias:@
.read_34_disablecopyonread_adam_m_output_kernel:@
.read_35_disablecopyonread_adam_v_output_kernel::
,read_36_disablecopyonread_adam_m_output_bias::
,read_37_disablecopyonread_adam_v_output_bias:)
read_38_disablecopyonread_total: )
read_39_disablecopyonread_count: 7
(read_40_disablecopyonread_true_positives:	�7
(read_41_disablecopyonread_true_negatives:	�8
)read_42_disablecopyonread_false_positives:	�8
)read_43_disablecopyonread_false_negatives:	�
savev2_const
identity_89��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_cdr3_conv_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_cdr3_conv_kernel^Read/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0m
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:e

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*"
_output_shapes
:{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_cdr3_conv_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_cdr3_conv_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_hla_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_hla_dense_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_hla_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_hla_dense_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_4/DisableCopyOnReadDisableCopyOnRead)read_4_disablecopyonread_tcr_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp)read_4_disablecopyonread_tcr_dense_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
@*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
@d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
@{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_tcr_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_tcr_dense_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_6/DisableCopyOnReadDisableCopyOnRead+read_6_disablecopyonread_pair_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp+read_6_disablecopyonread_pair_dense1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@}
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_pair_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_pair_dense1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_8/DisableCopyOnReadDisableCopyOnRead+read_8_disablecopyonread_pair_dense2_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp+read_8_disablecopyonread_pair_dense2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:@}
Read_9/DisableCopyOnReadDisableCopyOnRead)read_9_disablecopyonread_pair_dense2_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp)read_9_disablecopyonread_pair_dense2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_output_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_output_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_output_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_12/DisableCopyOnReadDisableCopyOnRead#read_12_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp#read_12_disablecopyonread_iteration^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_learning_rate^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnRead1read_14_disablecopyonread_adam_m_cdr3_conv_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp1read_14_disablecopyonread_adam_m_cdr3_conv_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead1read_15_disablecopyonread_adam_v_cdr3_conv_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp1read_15_disablecopyonread_adam_v_cdr3_conv_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:*
dtype0s
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:i
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*"
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead/read_16_disablecopyonread_adam_m_cdr3_conv_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp/read_16_disablecopyonread_adam_m_cdr3_conv_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_adam_v_cdr3_conv_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_adam_v_cdr3_conv_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead1read_18_disablecopyonread_adam_m_hla_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp1read_18_disablecopyonread_adam_m_hla_dense_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_19/DisableCopyOnReadDisableCopyOnRead1read_19_disablecopyonread_adam_v_hla_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp1read_19_disablecopyonread_adam_v_hla_dense_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_20/DisableCopyOnReadDisableCopyOnRead/read_20_disablecopyonread_adam_m_hla_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp/read_20_disablecopyonread_adam_m_hla_dense_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_adam_v_hla_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_adam_v_hla_dense_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_22/DisableCopyOnReadDisableCopyOnRead1read_22_disablecopyonread_adam_m_tcr_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp1read_22_disablecopyonread_adam_m_tcr_dense_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
@*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
@f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
@�
Read_23/DisableCopyOnReadDisableCopyOnRead1read_23_disablecopyonread_adam_v_tcr_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp1read_23_disablecopyonread_adam_v_tcr_dense_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�
@*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�
@f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
@�
Read_24/DisableCopyOnReadDisableCopyOnRead/read_24_disablecopyonread_adam_m_tcr_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp/read_24_disablecopyonread_adam_m_tcr_dense_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_25/DisableCopyOnReadDisableCopyOnRead/read_25_disablecopyonread_adam_v_tcr_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp/read_25_disablecopyonread_adam_v_tcr_dense_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_26/DisableCopyOnReadDisableCopyOnRead3read_26_disablecopyonread_adam_m_pair_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp3read_26_disablecopyonread_adam_m_pair_dense1_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_27/DisableCopyOnReadDisableCopyOnRead3read_27_disablecopyonread_adam_v_pair_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp3read_27_disablecopyonread_adam_v_pair_dense1_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_28/DisableCopyOnReadDisableCopyOnRead1read_28_disablecopyonread_adam_m_pair_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp1read_28_disablecopyonread_adam_m_pair_dense1_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_29/DisableCopyOnReadDisableCopyOnRead1read_29_disablecopyonread_adam_v_pair_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp1read_29_disablecopyonread_adam_v_pair_dense1_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_30/DisableCopyOnReadDisableCopyOnRead3read_30_disablecopyonread_adam_m_pair_dense2_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp3read_30_disablecopyonread_adam_m_pair_dense2_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_31/DisableCopyOnReadDisableCopyOnRead3read_31_disablecopyonread_adam_v_pair_dense2_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp3read_31_disablecopyonread_adam_v_pair_dense2_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_32/DisableCopyOnReadDisableCopyOnRead1read_32_disablecopyonread_adam_m_pair_dense2_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp1read_32_disablecopyonread_adam_m_pair_dense2_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_33/DisableCopyOnReadDisableCopyOnRead1read_33_disablecopyonread_adam_v_pair_dense2_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp1read_33_disablecopyonread_adam_v_pair_dense2_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnRead.read_34_disablecopyonread_adam_m_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp.read_34_disablecopyonread_adam_m_output_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_35/DisableCopyOnReadDisableCopyOnRead.read_35_disablecopyonread_adam_v_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp.read_35_disablecopyonread_adam_v_output_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_36/DisableCopyOnReadDisableCopyOnRead,read_36_disablecopyonread_adam_m_output_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp,read_36_disablecopyonread_adam_m_output_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_37/DisableCopyOnReadDisableCopyOnRead,read_37_disablecopyonread_adam_v_output_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp,read_37_disablecopyonread_adam_v_output_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_38/DisableCopyOnReadDisableCopyOnReadread_38_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpread_38_disablecopyonread_total^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_39/DisableCopyOnReadDisableCopyOnReadread_39_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpread_39_disablecopyonread_count^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_40/DisableCopyOnReadDisableCopyOnRead(read_40_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp(read_40_disablecopyonread_true_positives^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_41/DisableCopyOnReadDisableCopyOnRead(read_41_disablecopyonread_true_negatives"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp(read_41_disablecopyonread_true_negatives^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_42/DisableCopyOnReadDisableCopyOnRead)read_42_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp)read_42_disablecopyonread_false_positives^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_43/DisableCopyOnReadDisableCopyOnRead)read_43_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp)read_43_disablecopyonread_false_negatives^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *;
dtypes1
/2-	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_88Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_89IdentityIdentity_88:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_89Identity_89:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=-9

_output_shapes
: 

_user_specified_nameConst:/,+
)
_user_specified_namefalse_negatives:/++
)
_user_specified_namefalse_positives:.**
(
_user_specified_nametrue_negatives:.)*
(
_user_specified_nametrue_positives:%(!

_user_specified_namecount:%'!

_user_specified_nametotal:2&.
,
_user_specified_nameAdam/v/output/bias:2%.
,
_user_specified_nameAdam/m/output/bias:4$0
.
_user_specified_nameAdam/v/output/kernel:4#0
.
_user_specified_nameAdam/m/output/kernel:7"3
1
_user_specified_nameAdam/v/pair_dense2/bias:7!3
1
_user_specified_nameAdam/m/pair_dense2/bias:9 5
3
_user_specified_nameAdam/v/pair_dense2/kernel:95
3
_user_specified_nameAdam/m/pair_dense2/kernel:73
1
_user_specified_nameAdam/v/pair_dense1/bias:73
1
_user_specified_nameAdam/m/pair_dense1/bias:95
3
_user_specified_nameAdam/v/pair_dense1/kernel:95
3
_user_specified_nameAdam/m/pair_dense1/kernel:51
/
_user_specified_nameAdam/v/TCR_dense/bias:51
/
_user_specified_nameAdam/m/TCR_dense/bias:73
1
_user_specified_nameAdam/v/TCR_dense/kernel:73
1
_user_specified_nameAdam/m/TCR_dense/kernel:51
/
_user_specified_nameAdam/v/HLA_dense/bias:51
/
_user_specified_nameAdam/m/HLA_dense/bias:73
1
_user_specified_nameAdam/v/HLA_dense/kernel:73
1
_user_specified_nameAdam/m/HLA_dense/kernel:51
/
_user_specified_nameAdam/v/CDR3_conv/bias:51
/
_user_specified_nameAdam/m/CDR3_conv/bias:73
1
_user_specified_nameAdam/v/CDR3_conv/kernel:73
1
_user_specified_nameAdam/m/CDR3_conv/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:+'
%
_user_specified_nameoutput/bias:-)
'
_user_specified_nameoutput/kernel:0
,
*
_user_specified_namepair_dense2/bias:2	.
,
_user_specified_namepair_dense2/kernel:0,
*
_user_specified_namepair_dense1/bias:2.
,
_user_specified_namepair_dense1/kernel:.*
(
_user_specified_nameTCR_dense/bias:0,
*
_user_specified_nameTCR_dense/kernel:.*
(
_user_specified_nameHLA_dense/bias:0,
*
_user_specified_nameHLA_dense/kernel:.*
(
_user_specified_nameCDR3_conv/bias:0,
*
_user_specified_nameCDR3_conv/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
G__inference_pair_dense2_layer_call_and_return_conditional_losses_355871

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
f
J__inference_CDR2.5_reshape_layer_call_and_return_conditional_losses_355766

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������~X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������~"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
d
H__inference_CDR3_reshape_layer_call_and_return_conditional_losses_355719

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
f
J__inference_CDR2.5_reshape_layer_call_and_return_conditional_losses_356299

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������~X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������~"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_concatenate_1_layer_call_and_return_conditional_losses_355777

inputs
inputs_1
inputs_2
inputs_3
inputs_4
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*(
_output_shapes
:����������
X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:���������:����������:����������:����������:���������~:OK
'
_output_shapes
:���������~
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
X
,__inference_concatenate_layer_call_fn_356241
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_355727a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs_0
�

�
B__inference_output_layer_call_and_return_conditional_losses_356475

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
.__inference_concatenate_1_layer_call_fn_356325
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_355777a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:���������:����������:����������:����������:���������~:QM
'
_output_shapes
:���������~
"
_user_specified_name
inputs_4:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_3:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_2:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
ω
�
!__inference__wrapped_model_355658
hla
cdr3
cdr3_len
cdr1
cdr2

cdr2_5Q
;model_cdr3_conv_conv1d_expanddims_1_readvariableop_resource:=
/model_cdr3_conv_biasadd_readvariableop_resource:A
.model_hla_dense_matmul_readvariableop_resource:	�@=
/model_hla_dense_biasadd_readvariableop_resource:@A
.model_tcr_dense_matmul_readvariableop_resource:	�
@=
/model_tcr_dense_biasadd_readvariableop_resource:@C
0model_pair_dense1_matmul_readvariableop_resource:	�@?
1model_pair_dense1_biasadd_readvariableop_resource:@B
0model_pair_dense2_matmul_readvariableop_resource:@?
1model_pair_dense2_biasadd_readvariableop_resource:=
+model_output_matmul_readvariableop_resource::
,model_output_biasadd_readvariableop_resource:
identity��&model/CDR3_conv/BiasAdd/ReadVariableOp�2model/CDR3_conv/Conv1D/ExpandDims_1/ReadVariableOp�&model/HLA_dense/BiasAdd/ReadVariableOp�%model/HLA_dense/MatMul/ReadVariableOp�&model/TCR_dense/BiasAdd/ReadVariableOp�%model/TCR_dense/MatMul/ReadVariableOp�#model/output/BiasAdd/ReadVariableOp�"model/output/MatMul/ReadVariableOp�(model/pair_dense1/BiasAdd/ReadVariableOp�'model/pair_dense1/MatMul/ReadVariableOp�(model/pair_dense2/BiasAdd/ReadVariableOp�'model/pair_dense2/MatMul/ReadVariableOpp
%model/CDR3_conv/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
!model/CDR3_conv/Conv1D/ExpandDims
ExpandDimscdr3.model/CDR3_conv/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
2model/CDR3_conv/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp;model_cdr3_conv_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0i
'model/CDR3_conv/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
#model/CDR3_conv/Conv1D/ExpandDims_1
ExpandDims:model/CDR3_conv/Conv1D/ExpandDims_1/ReadVariableOp:value:00model/CDR3_conv/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
model/CDR3_conv/Conv1DConv2D*model/CDR3_conv/Conv1D/ExpandDims:output:0,model/CDR3_conv/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
model/CDR3_conv/Conv1D/SqueezeSqueezemodel/CDR3_conv/Conv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
&model/CDR3_conv/BiasAdd/ReadVariableOpReadVariableOp/model_cdr3_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/CDR3_conv/BiasAddBiasAdd'model/CDR3_conv/Conv1D/Squeeze:output:0.model/CDR3_conv/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������t
model/CDR3_conv/ReluRelu model/CDR3_conv/BiasAdd:output:0*
T0*+
_output_shapes
:���������f
$model/CDR3_maxpooling/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
 model/CDR3_maxpooling/ExpandDims
ExpandDims"model/CDR3_conv/Relu:activations:0-model/CDR3_maxpooling/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
model/CDR3_maxpooling/MaxPoolMaxPool)model/CDR3_maxpooling/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
model/CDR3_maxpooling/SqueezeSqueeze&model/CDR3_maxpooling/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
i
model/CDR3_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
model/CDR3_flatten/ReshapeReshape&model/CDR3_maxpooling/Squeeze:output:0!model/CDR3_flatten/Const:output:0*
T0*(
_output_shapes
:����������Z
model/CDR3_reshape/ShapeShapecdr3*
T0*
_output_shapes
::��p
&model/CDR3_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(model/CDR3_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model/CDR3_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model/CDR3_reshape/strided_sliceStridedSlice!model/CDR3_reshape/Shape:output:0/model/CDR3_reshape/strided_slice/stack:output:01model/CDR3_reshape/strided_slice/stack_1:output:01model/CDR3_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"model/CDR3_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
 model/CDR3_reshape/Reshape/shapePack)model/CDR3_reshape/strided_slice:output:0+model/CDR3_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/CDR3_reshape/ReshapeReshapecdr3)model/CDR3_reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:����������_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2#model/CDR3_flatten/Reshape:output:0#model/CDR3_reshape/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������Z
model/CDR1_reshape/ShapeShapecdr1*
T0*
_output_shapes
::��p
&model/CDR1_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(model/CDR1_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model/CDR1_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model/CDR1_reshape/strided_sliceStridedSlice!model/CDR1_reshape/Shape:output:0/model/CDR1_reshape/strided_slice/stack:output:01model/CDR1_reshape/strided_slice/stack_1:output:01model/CDR1_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"model/CDR1_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
 model/CDR1_reshape/Reshape/shapePack)model/CDR1_reshape/strided_slice:output:0+model/CDR1_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/CDR1_reshape/ReshapeReshapecdr1)model/CDR1_reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:����������Z
model/CDR2_reshape/ShapeShapecdr2*
T0*
_output_shapes
::��p
&model/CDR2_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(model/CDR2_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(model/CDR2_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 model/CDR2_reshape/strided_sliceStridedSlice!model/CDR2_reshape/Shape:output:0/model/CDR2_reshape/strided_slice/stack:output:01model/CDR2_reshape/strided_slice/stack_1:output:01model/CDR2_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"model/CDR2_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
 model/CDR2_reshape/Reshape/shapePack)model/CDR2_reshape/strided_slice:output:0+model/CDR2_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/CDR2_reshape/ReshapeReshapecdr2)model/CDR2_reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:����������^
model/CDR2.5_reshape/ShapeShapecdr2_5*
T0*
_output_shapes
::��r
(model/CDR2.5_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*model/CDR2.5_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*model/CDR2.5_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"model/CDR2.5_reshape/strided_sliceStridedSlice#model/CDR2.5_reshape/Shape:output:01model/CDR2.5_reshape/strided_slice/stack:output:03model/CDR2.5_reshape/strided_slice/stack_1:output:03model/CDR2.5_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$model/CDR2.5_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~�
"model/CDR2.5_reshape/Reshape/shapePack+model/CDR2.5_reshape/strided_slice:output:0-model/CDR2.5_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/CDR2.5_reshape/ReshapeReshapecdr2_5+model/CDR2.5_reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:���������~a
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate_1/concatConcatV2cdr3_len!model/concatenate/concat:output:0#model/CDR1_reshape/Reshape:output:0#model/CDR2_reshape/Reshape:output:0%model/CDR2.5_reshape/Reshape:output:0(model/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������
X
model/HLA_reshape/ShapeShapehla*
T0*
_output_shapes
::��o
%model/HLA_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model/HLA_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model/HLA_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/HLA_reshape/strided_sliceStridedSlice model/HLA_reshape/Shape:output:0.model/HLA_reshape/strided_slice/stack:output:00model/HLA_reshape/strided_slice/stack_1:output:00model/HLA_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
!model/HLA_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
model/HLA_reshape/Reshape/shapePack(model/HLA_reshape/strided_slice:output:0*model/HLA_reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/HLA_reshape/ReshapeReshapehla(model/HLA_reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
%model/HLA_dense/MatMul/ReadVariableOpReadVariableOp.model_hla_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/HLA_dense/MatMulMatMul"model/HLA_reshape/Reshape:output:0-model/HLA_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&model/HLA_dense/BiasAdd/ReadVariableOpReadVariableOp/model_hla_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/HLA_dense/BiasAddBiasAdd model/HLA_dense/MatMul:product:0.model/HLA_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
model/HLA_dense/ReluRelu model/HLA_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
%model/TCR_dense/MatMul/ReadVariableOpReadVariableOp.model_tcr_dense_matmul_readvariableop_resource*
_output_shapes
:	�
@*
dtype0�
model/TCR_dense/MatMulMatMul#model/concatenate_1/concat:output:0-model/TCR_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
&model/TCR_dense/BiasAdd/ReadVariableOpReadVariableOp/model_tcr_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/TCR_dense/BiasAddBiasAdd model/TCR_dense/MatMul:product:0.model/TCR_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
model/TCR_dense/ReluRelu model/TCR_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@a
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate_2/concatConcatV2"model/HLA_dense/Relu:activations:0"model/TCR_dense/Relu:activations:0(model/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
'model/pair_dense1/MatMul/ReadVariableOpReadVariableOp0model_pair_dense1_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/pair_dense1/MatMulMatMul#model/concatenate_2/concat:output:0/model/pair_dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
(model/pair_dense1/BiasAdd/ReadVariableOpReadVariableOp1model_pair_dense1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/pair_dense1/BiasAddBiasAdd"model/pair_dense1/MatMul:product:00model/pair_dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@t
model/pair_dense1/ReluRelu"model/pair_dense1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@z
model/dropout/IdentityIdentity$model/pair_dense1/Relu:activations:0*
T0*'
_output_shapes
:���������@�
'model/pair_dense2/MatMul/ReadVariableOpReadVariableOp0model_pair_dense2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/pair_dense2/MatMulMatMulmodel/dropout/Identity:output:0/model/pair_dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(model/pair_dense2/BiasAdd/ReadVariableOpReadVariableOp1model_pair_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/pair_dense2/BiasAddBiasAdd"model/pair_dense2/MatMul:product:00model/pair_dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
model/pair_dense2/ReluRelu"model/pair_dense2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model/output/MatMulMatMul$model/pair_dense2/Relu:activations:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
model/output/SigmoidSigmoidmodel/output/BiasAdd:output:0*
T0*'
_output_shapes
:���������g
IdentityIdentitymodel/output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^model/CDR3_conv/BiasAdd/ReadVariableOp3^model/CDR3_conv/Conv1D/ExpandDims_1/ReadVariableOp'^model/HLA_dense/BiasAdd/ReadVariableOp&^model/HLA_dense/MatMul/ReadVariableOp'^model/TCR_dense/BiasAdd/ReadVariableOp&^model/TCR_dense/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp)^model/pair_dense1/BiasAdd/ReadVariableOp(^model/pair_dense1/MatMul/ReadVariableOp)^model/pair_dense2/BiasAdd/ReadVariableOp(^model/pair_dense2/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2P
&model/CDR3_conv/BiasAdd/ReadVariableOp&model/CDR3_conv/BiasAdd/ReadVariableOp2h
2model/CDR3_conv/Conv1D/ExpandDims_1/ReadVariableOp2model/CDR3_conv/Conv1D/ExpandDims_1/ReadVariableOp2P
&model/HLA_dense/BiasAdd/ReadVariableOp&model/HLA_dense/BiasAdd/ReadVariableOp2N
%model/HLA_dense/MatMul/ReadVariableOp%model/HLA_dense/MatMul/ReadVariableOp2P
&model/TCR_dense/BiasAdd/ReadVariableOp&model/TCR_dense/BiasAdd/ReadVariableOp2N
%model/TCR_dense/MatMul/ReadVariableOp%model/TCR_dense/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp2T
(model/pair_dense1/BiasAdd/ReadVariableOp(model/pair_dense1/BiasAdd/ReadVariableOp2R
'model/pair_dense1/MatMul/ReadVariableOp'model/pair_dense1/MatMul/ReadVariableOp2T
(model/pair_dense2/BiasAdd/ReadVariableOp(model/pair_dense2/BiasAdd/ReadVariableOp2R
'model/pair_dense2/MatMul/ReadVariableOp'model/pair_dense2/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:SO
+
_output_shapes
:���������
 
_user_specified_nameCDR2.5:QM
+
_output_shapes
:���������


_user_specified_nameCDR2:QM
+
_output_shapes
:���������

_user_specified_nameCDR1:QM
'
_output_shapes
:���������
"
_user_specified_name
CDR3_len:QM
+
_output_shapes
:���������

_user_specified_nameCDR3:P L
+
_output_shapes
:���������-

_user_specified_nameHLA
�
�
I__inference_concatenate_1_layer_call_and_return_conditional_losses_356335
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*(
_output_shapes
:����������
X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:���������:����������:����������:����������:���������~:QM
'
_output_shapes
:���������~
"
_user_specified_name
inputs_4:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_3:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_2:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�	
d
H__inference_CDR1_reshape_layer_call_and_return_conditional_losses_356265

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_CDR3_conv_layer_call_and_return_conditional_losses_355694

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
H__inference_CDR3_flatten_layer_call_and_return_conditional_losses_356218

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_356169
cdr1
cdr2

cdr2_5
cdr3
cdr3_len
hla
unknown:
	unknown_0:
	unknown_1:	�@
	unknown_2:@
	unknown_3:	�
@
	unknown_4:@
	unknown_5:	�@
	unknown_6:@
	unknown_7:@
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallhlacdr3cdr3_lencdr1cdr2cdr2_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_355658o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������
:���������:���������:���������:���������-: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name356165:&"
 
_user_specified_name356163:&"
 
_user_specified_name356161:&"
 
_user_specified_name356159:&"
 
_user_specified_name356157:&"
 
_user_specified_name356155:&"
 
_user_specified_name356153:&
"
 
_user_specified_name356151:&	"
 
_user_specified_name356149:&"
 
_user_specified_name356147:&"
 
_user_specified_name356145:&"
 
_user_specified_name356143:PL
+
_output_shapes
:���������-

_user_specified_nameHLA:QM
'
_output_shapes
:���������
"
_user_specified_name
CDR3_len:QM
+
_output_shapes
:���������

_user_specified_nameCDR3:SO
+
_output_shapes
:���������
 
_user_specified_nameCDR2.5:QM
+
_output_shapes
:���������


_user_specified_nameCDR2:Q M
+
_output_shapes
:���������

_user_specified_nameCDR1
�
�
'__inference_output_layer_call_fn_356464

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_355887o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name356460:&"
 
_user_specified_name356458:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_pair_dense2_layer_call_fn_356444

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_pair_dense2_layer_call_and_return_conditional_losses_355871o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name356440:&"
 
_user_specified_name356438:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
Z
.__inference_concatenate_2_layer_call_fn_356381
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_355830a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0
�	
c
G__inference_HLA_reshape_layer_call_and_return_conditional_losses_355790

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-:S O
+
_output_shapes
:���������-
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_355859

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�>[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
G__inference_pair_dense1_layer_call_and_return_conditional_losses_356408

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
d
H__inference_CDR1_reshape_layer_call_and_return_conditional_losses_355740

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_TCR_dense_layer_call_and_return_conditional_losses_355818

inputs1
matmul_readvariableop_resource:	�
@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�
g
K__inference_CDR3_maxpooling_layer_call_and_return_conditional_losses_356207

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+����������������������������
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_355936

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
a
(__inference_dropout_layer_call_fn_356413

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_355859o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_TCR_dense_layer_call_fn_356364

inputs
unknown:	�
@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_TCR_dense_layer_call_and_return_conditional_losses_355818o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name356360:&"
 
_user_specified_name356358:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�	
d
H__inference_CDR3_reshape_layer_call_and_return_conditional_losses_356235

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
s
I__inference_concatenate_2_layer_call_and_return_conditional_losses_355830

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
L
0__inference_CDR3_maxpooling_layer_call_fn_356199

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_CDR3_maxpooling_layer_call_and_return_conditional_losses_355666v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�I
�
A__inference_model_layer_call_and_return_conditional_losses_355894
hla
cdr3
cdr3_len
cdr1
cdr2

cdr2_5&
cdr3_conv_355695:
cdr3_conv_355697:#
hla_dense_355803:	�@
hla_dense_355805:@#
tcr_dense_355819:	�
@
tcr_dense_355821:@%
pair_dense1_355843:	�@ 
pair_dense1_355845:@$
pair_dense2_355872:@ 
pair_dense2_355874:
output_355888:
output_355890:
identity��!CDR3_conv/StatefulPartitionedCall�!HLA_dense/StatefulPartitionedCall�!TCR_dense/StatefulPartitionedCall�dropout/StatefulPartitionedCall�output/StatefulPartitionedCall�#pair_dense1/StatefulPartitionedCall�#pair_dense2/StatefulPartitionedCall�
!CDR3_conv/StatefulPartitionedCallStatefulPartitionedCallcdr3cdr3_conv_355695cdr3_conv_355697*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_CDR3_conv_layer_call_and_return_conditional_losses_355694�
CDR3_maxpooling/PartitionedCallPartitionedCall*CDR3_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_CDR3_maxpooling_layer_call_and_return_conditional_losses_355666�
CDR3_flatten/PartitionedCallPartitionedCall(CDR3_maxpooling/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR3_flatten_layer_call_and_return_conditional_losses_355706�
CDR3_reshape/PartitionedCallPartitionedCallcdr3*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR3_reshape_layer_call_and_return_conditional_losses_355719�
concatenate/PartitionedCallPartitionedCall%CDR3_flatten/PartitionedCall:output:0%CDR3_reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_355727�
CDR1_reshape/PartitionedCallPartitionedCallcdr1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR1_reshape_layer_call_and_return_conditional_losses_355740�
CDR2_reshape/PartitionedCallPartitionedCallcdr2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR2_reshape_layer_call_and_return_conditional_losses_355753�
CDR2.5_reshape/PartitionedCallPartitionedCallcdr2_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������~* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_CDR2.5_reshape_layer_call_and_return_conditional_losses_355766�
concatenate_1/PartitionedCallPartitionedCallcdr3_len$concatenate/PartitionedCall:output:0%CDR1_reshape/PartitionedCall:output:0%CDR2_reshape/PartitionedCall:output:0'CDR2.5_reshape/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_1_layer_call_and_return_conditional_losses_355777�
HLA_reshape/PartitionedCallPartitionedCallhla*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_HLA_reshape_layer_call_and_return_conditional_losses_355790�
!HLA_dense/StatefulPartitionedCallStatefulPartitionedCall$HLA_reshape/PartitionedCall:output:0hla_dense_355803hla_dense_355805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_HLA_dense_layer_call_and_return_conditional_losses_355802�
!TCR_dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0tcr_dense_355819tcr_dense_355821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_TCR_dense_layer_call_and_return_conditional_losses_355818�
concatenate_2/PartitionedCallPartitionedCall*HLA_dense/StatefulPartitionedCall:output:0*TCR_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_355830�
#pair_dense1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0pair_dense1_355843pair_dense1_355845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_pair_dense1_layer_call_and_return_conditional_losses_355842�
dropout/StatefulPartitionedCallStatefulPartitionedCall,pair_dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_355859�
#pair_dense2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0pair_dense2_355872pair_dense2_355874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_pair_dense2_layer_call_and_return_conditional_losses_355871�
output/StatefulPartitionedCallStatefulPartitionedCall,pair_dense2/StatefulPartitionedCall:output:0output_355888output_355890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_355887v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^CDR3_conv/StatefulPartitionedCall"^HLA_dense/StatefulPartitionedCall"^TCR_dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^output/StatefulPartitionedCall$^pair_dense1/StatefulPartitionedCall$^pair_dense2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2F
!CDR3_conv/StatefulPartitionedCall!CDR3_conv/StatefulPartitionedCall2F
!HLA_dense/StatefulPartitionedCall!HLA_dense/StatefulPartitionedCall2F
!TCR_dense/StatefulPartitionedCall!TCR_dense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall2J
#pair_dense1/StatefulPartitionedCall#pair_dense1/StatefulPartitionedCall2J
#pair_dense2/StatefulPartitionedCall#pair_dense2/StatefulPartitionedCall:&"
 
_user_specified_name355890:&"
 
_user_specified_name355888:&"
 
_user_specified_name355874:&"
 
_user_specified_name355872:&"
 
_user_specified_name355845:&"
 
_user_specified_name355843:&"
 
_user_specified_name355821:&
"
 
_user_specified_name355819:&	"
 
_user_specified_name355805:&"
 
_user_specified_name355803:&"
 
_user_specified_name355697:&"
 
_user_specified_name355695:SO
+
_output_shapes
:���������
 
_user_specified_nameCDR2.5:QM
+
_output_shapes
:���������


_user_specified_nameCDR2:QM
+
_output_shapes
:���������

_user_specified_nameCDR1:QM
'
_output_shapes
:���������
"
_user_specified_name
CDR3_len:QM
+
_output_shapes
:���������

_user_specified_nameCDR3:P L
+
_output_shapes
:���������-

_user_specified_nameHLA
�
D
(__inference_dropout_layer_call_fn_356418

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_355936`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
G__inference_pair_dense2_layer_call_and_return_conditional_losses_356455

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_pair_dense1_layer_call_fn_356397

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_pair_dense1_layer_call_and_return_conditional_losses_355842o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name356393:&"
 
_user_specified_name356391:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_HLA_reshape_layer_call_fn_356304

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_HLA_reshape_layer_call_and_return_conditional_losses_355790a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-:S O
+
_output_shapes
:���������-
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_356430

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�>[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_CDR3_conv_layer_call_fn_356178

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_CDR3_conv_layer_call_and_return_conditional_losses_355694s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name356174:&"
 
_user_specified_name356172:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_HLA_dense_layer_call_and_return_conditional_losses_355802

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
I
-__inference_CDR3_reshape_layer_call_fn_356223

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_CDR3_reshape_layer_call_and_return_conditional_losses_355719a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_CDR3_conv_layer_call_and_return_conditional_losses_356194

inputsA
+conv1d_expanddims_1_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_356907
file_prefix7
!assignvariableop_cdr3_conv_kernel:/
!assignvariableop_1_cdr3_conv_bias:6
#assignvariableop_2_hla_dense_kernel:	�@/
!assignvariableop_3_hla_dense_bias:@6
#assignvariableop_4_tcr_dense_kernel:	�
@/
!assignvariableop_5_tcr_dense_bias:@8
%assignvariableop_6_pair_dense1_kernel:	�@1
#assignvariableop_7_pair_dense1_bias:@7
%assignvariableop_8_pair_dense2_kernel:@1
#assignvariableop_9_pair_dense2_bias:3
!assignvariableop_10_output_kernel:-
assignvariableop_11_output_bias:'
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: A
+assignvariableop_14_adam_m_cdr3_conv_kernel:A
+assignvariableop_15_adam_v_cdr3_conv_kernel:7
)assignvariableop_16_adam_m_cdr3_conv_bias:7
)assignvariableop_17_adam_v_cdr3_conv_bias:>
+assignvariableop_18_adam_m_hla_dense_kernel:	�@>
+assignvariableop_19_adam_v_hla_dense_kernel:	�@7
)assignvariableop_20_adam_m_hla_dense_bias:@7
)assignvariableop_21_adam_v_hla_dense_bias:@>
+assignvariableop_22_adam_m_tcr_dense_kernel:	�
@>
+assignvariableop_23_adam_v_tcr_dense_kernel:	�
@7
)assignvariableop_24_adam_m_tcr_dense_bias:@7
)assignvariableop_25_adam_v_tcr_dense_bias:@@
-assignvariableop_26_adam_m_pair_dense1_kernel:	�@@
-assignvariableop_27_adam_v_pair_dense1_kernel:	�@9
+assignvariableop_28_adam_m_pair_dense1_bias:@9
+assignvariableop_29_adam_v_pair_dense1_bias:@?
-assignvariableop_30_adam_m_pair_dense2_kernel:@?
-assignvariableop_31_adam_v_pair_dense2_kernel:@9
+assignvariableop_32_adam_m_pair_dense2_bias:9
+assignvariableop_33_adam_v_pair_dense2_bias::
(assignvariableop_34_adam_m_output_kernel::
(assignvariableop_35_adam_v_output_kernel:4
&assignvariableop_36_adam_m_output_bias:4
&assignvariableop_37_adam_v_output_bias:#
assignvariableop_38_total: #
assignvariableop_39_count: 1
"assignvariableop_40_true_positives:	�1
"assignvariableop_41_true_negatives:	�2
#assignvariableop_42_false_positives:	�2
#assignvariableop_43_false_negatives:	�
identity_45��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_cdr3_conv_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_cdr3_conv_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_hla_dense_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_hla_dense_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_tcr_dense_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_tcr_dense_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_pair_dense1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_pair_dense1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_pair_dense2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_pair_dense2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_output_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_output_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterationIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp+assignvariableop_14_adam_m_cdr3_conv_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_v_cdr3_conv_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_m_cdr3_conv_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_v_cdr3_conv_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_m_hla_dense_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_v_hla_dense_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_m_hla_dense_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_v_hla_dense_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_m_tcr_dense_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_v_tcr_dense_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_m_tcr_dense_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_v_tcr_dense_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp-assignvariableop_26_adam_m_pair_dense1_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_v_pair_dense1_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_m_pair_dense1_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_v_pair_dense1_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_m_pair_dense2_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_v_pair_dense2_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_m_pair_dense2_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_v_pair_dense2_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_m_output_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_v_output_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_m_output_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp&assignvariableop_37_adam_v_output_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_totalIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_countIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp"assignvariableop_40_true_positivesIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp"assignvariableop_41_true_negativesIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp#assignvariableop_42_false_positivesIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp#assignvariableop_43_false_negativesIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_45IdentityIdentity_44:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_45Identity_45:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:/,+
)
_user_specified_namefalse_negatives:/++
)
_user_specified_namefalse_positives:.**
(
_user_specified_nametrue_negatives:.)*
(
_user_specified_nametrue_positives:%(!

_user_specified_namecount:%'!

_user_specified_nametotal:2&.
,
_user_specified_nameAdam/v/output/bias:2%.
,
_user_specified_nameAdam/m/output/bias:4$0
.
_user_specified_nameAdam/v/output/kernel:4#0
.
_user_specified_nameAdam/m/output/kernel:7"3
1
_user_specified_nameAdam/v/pair_dense2/bias:7!3
1
_user_specified_nameAdam/m/pair_dense2/bias:9 5
3
_user_specified_nameAdam/v/pair_dense2/kernel:95
3
_user_specified_nameAdam/m/pair_dense2/kernel:73
1
_user_specified_nameAdam/v/pair_dense1/bias:73
1
_user_specified_nameAdam/m/pair_dense1/bias:95
3
_user_specified_nameAdam/v/pair_dense1/kernel:95
3
_user_specified_nameAdam/m/pair_dense1/kernel:51
/
_user_specified_nameAdam/v/TCR_dense/bias:51
/
_user_specified_nameAdam/m/TCR_dense/bias:73
1
_user_specified_nameAdam/v/TCR_dense/kernel:73
1
_user_specified_nameAdam/m/TCR_dense/kernel:51
/
_user_specified_nameAdam/v/HLA_dense/bias:51
/
_user_specified_nameAdam/m/HLA_dense/bias:73
1
_user_specified_nameAdam/v/HLA_dense/kernel:73
1
_user_specified_nameAdam/m/HLA_dense/kernel:51
/
_user_specified_nameAdam/v/CDR3_conv/bias:51
/
_user_specified_nameAdam/m/CDR3_conv/bias:73
1
_user_specified_nameAdam/v/CDR3_conv/kernel:73
1
_user_specified_nameAdam/m/CDR3_conv/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:+'
%
_user_specified_nameoutput/bias:-)
'
_user_specified_nameoutput/kernel:0
,
*
_user_specified_namepair_dense2/bias:2	.
,
_user_specified_namepair_dense2/kernel:0,
*
_user_specified_namepair_dense1/bias:2.
,
_user_specified_namepair_dense1/kernel:.*
(
_user_specified_nameTCR_dense/bias:0,
*
_user_specified_nameTCR_dense/kernel:.*
(
_user_specified_nameHLA_dense/bias:0,
*
_user_specified_nameHLA_dense/kernel:.*
(
_user_specified_nameCDR3_conv/bias:0,
*
_user_specified_nameCDR3_conv/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
u
I__inference_concatenate_2_layer_call_and_return_conditional_losses_356388
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0
�

�
B__inference_output_layer_call_and_return_conditional_losses_355887

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
q
G__inference_concatenate_layer_call_and_return_conditional_losses_355727

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
CDR11
serving_default_CDR1:0���������
=
CDR2.53
serving_default_CDR2.5:0���������
9
CDR21
serving_default_CDR2:0���������

=
CDR3_len1
serving_default_CDR3_len:0���������
9
CDR31
serving_default_CDR3:0���������
7
HLA0
serving_default_HLA:0���������-:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-1
layer-16
layer_with_weights-2
layer-17
layer-18
layer_with_weights-3
layer-19
layer-20
layer_with_weights-4
layer-21
layer_with_weights-5
layer-22
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
 
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op"
_tf_keras_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

nkernel
obias"
_tf_keras_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias"
_tf_keras_layer
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
z
'0
(1
f2
g3
n4
o5
|6
}7
�8
�9
�10
�11"
trackable_list_wrapper
z
'0
(1
f2
g3
n4
o5
|6
}7
�8
�9
�10
�11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_model_layer_call_fn_355983
&__inference_model_layer_call_fn_356017�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_model_layer_call_and_return_conditional_losses_355894
A__inference_model_layer_call_and_return_conditional_losses_355949�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_355658HLACDR3CDR3_lenCDR1CDR2CDR2.5"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_CDR3_conv_layer_call_fn_356178�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_CDR3_conv_layer_call_and_return_conditional_losses_356194�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
&:$2CDR3_conv/kernel
:2CDR3_conv/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_CDR3_maxpooling_layer_call_fn_356199�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_CDR3_maxpooling_layer_call_and_return_conditional_losses_356207�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_CDR3_flatten_layer_call_fn_356212�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_CDR3_flatten_layer_call_and_return_conditional_losses_356218�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_CDR3_reshape_layer_call_fn_356223�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_CDR3_reshape_layer_call_and_return_conditional_losses_356235�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_concatenate_layer_call_fn_356241�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_layer_call_and_return_conditional_losses_356248�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_CDR1_reshape_layer_call_fn_356253�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_CDR1_reshape_layer_call_and_return_conditional_losses_356265�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_CDR2_reshape_layer_call_fn_356270�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_CDR2_reshape_layer_call_and_return_conditional_losses_356282�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_CDR2.5_reshape_layer_call_fn_356287�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_CDR2.5_reshape_layer_call_and_return_conditional_losses_356299�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_HLA_reshape_layer_call_fn_356304�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_HLA_reshape_layer_call_and_return_conditional_losses_356316�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_1_layer_call_fn_356325�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_1_layer_call_and_return_conditional_losses_356335�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_HLA_dense_layer_call_fn_356344�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_HLA_dense_layer_call_and_return_conditional_losses_356355�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!	�@2HLA_dense/kernel
:@2HLA_dense/bias
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_TCR_dense_layer_call_fn_356364�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_TCR_dense_layer_call_and_return_conditional_losses_356375�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!	�
@2TCR_dense/kernel
:@2TCR_dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_2_layer_call_fn_356381�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_356388�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_pair_dense1_layer_call_fn_356397�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_pair_dense1_layer_call_and_return_conditional_losses_356408�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#	�@2pair_dense1/kernel
:@2pair_dense1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_layer_call_fn_356413
(__inference_dropout_layer_call_fn_356418�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_layer_call_and_return_conditional_losses_356430
C__inference_dropout_layer_call_and_return_conditional_losses_356435�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_pair_dense2_layer_call_fn_356444�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_pair_dense2_layer_call_and_return_conditional_losses_356455�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"@2pair_dense2/kernel
:2pair_dense2/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_output_layer_call_fn_356464�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_output_layer_call_and_return_conditional_losses_356475�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:2output/kernel
:2output/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_model_layer_call_fn_355983HLACDR3CDR3_lenCDR1CDR2CDR2.5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_model_layer_call_fn_356017HLACDR3CDR3_lenCDR1CDR2CDR2.5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_355894HLACDR3CDR3_lenCDR1CDR2CDR2.5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_355949HLACDR3CDR3_lenCDR1CDR2CDR2.5"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_356169CDR1CDR2CDR2.5CDR3CDR3_lenHLA"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 F

kwonlyargs8�5
jCDR1
jCDR2
jCDR2_5
jCDR3

jCDR3_len
jHLA
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_CDR3_conv_layer_call_fn_356178inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_CDR3_conv_layer_call_and_return_conditional_losses_356194inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_CDR3_maxpooling_layer_call_fn_356199inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_CDR3_maxpooling_layer_call_and_return_conditional_losses_356207inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_CDR3_flatten_layer_call_fn_356212inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_CDR3_flatten_layer_call_and_return_conditional_losses_356218inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_CDR3_reshape_layer_call_fn_356223inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_CDR3_reshape_layer_call_and_return_conditional_losses_356235inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_concatenate_layer_call_fn_356241inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_concatenate_layer_call_and_return_conditional_losses_356248inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_CDR1_reshape_layer_call_fn_356253inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_CDR1_reshape_layer_call_and_return_conditional_losses_356265inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_CDR2_reshape_layer_call_fn_356270inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_CDR2_reshape_layer_call_and_return_conditional_losses_356282inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_CDR2.5_reshape_layer_call_fn_356287inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_CDR2.5_reshape_layer_call_and_return_conditional_losses_356299inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_HLA_reshape_layer_call_fn_356304inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_HLA_reshape_layer_call_and_return_conditional_losses_356316inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_concatenate_1_layer_call_fn_356325inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_1_layer_call_and_return_conditional_losses_356335inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_HLA_dense_layer_call_fn_356344inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_HLA_dense_layer_call_and_return_conditional_losses_356355inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_TCR_dense_layer_call_fn_356364inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_TCR_dense_layer_call_and_return_conditional_losses_356375inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_concatenate_2_layer_call_fn_356381inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_356388inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_pair_dense1_layer_call_fn_356397inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_pair_dense1_layer_call_and_return_conditional_losses_356408inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dropout_layer_call_fn_356413inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_layer_call_fn_356418inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_356430inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_356435inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_pair_dense2_layer_call_fn_356444inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_pair_dense2_layer_call_and_return_conditional_losses_356455inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_output_layer_call_fn_356464inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_output_layer_call_and_return_conditional_losses_356475inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives"
_tf_keras_metric
+:)2Adam/m/CDR3_conv/kernel
+:)2Adam/v/CDR3_conv/kernel
!:2Adam/m/CDR3_conv/bias
!:2Adam/v/CDR3_conv/bias
(:&	�@2Adam/m/HLA_dense/kernel
(:&	�@2Adam/v/HLA_dense/kernel
!:@2Adam/m/HLA_dense/bias
!:@2Adam/v/HLA_dense/bias
(:&	�
@2Adam/m/TCR_dense/kernel
(:&	�
@2Adam/v/TCR_dense/kernel
!:@2Adam/m/TCR_dense/bias
!:@2Adam/v/TCR_dense/bias
*:(	�@2Adam/m/pair_dense1/kernel
*:(	�@2Adam/v/pair_dense1/kernel
#:!@2Adam/m/pair_dense1/bias
#:!@2Adam/v/pair_dense1/bias
):'@2Adam/m/pair_dense2/kernel
):'@2Adam/v/pair_dense2/kernel
#:!2Adam/m/pair_dense2/bias
#:!2Adam/v/pair_dense2/bias
$:"2Adam/m/output/kernel
$:"2Adam/v/output/kernel
:2Adam/m/output/bias
:2Adam/v/output/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives�
H__inference_CDR1_reshape_layer_call_and_return_conditional_losses_356265d3�0
)�&
$�!
inputs���������
� "-�*
#� 
tensor_0����������
� �
-__inference_CDR1_reshape_layer_call_fn_356253Y3�0
)�&
$�!
inputs���������
� ""�
unknown�����������
J__inference_CDR2.5_reshape_layer_call_and_return_conditional_losses_356299c3�0
)�&
$�!
inputs���������
� ",�)
"�
tensor_0���������~
� �
/__inference_CDR2.5_reshape_layer_call_fn_356287X3�0
)�&
$�!
inputs���������
� "!�
unknown���������~�
H__inference_CDR2_reshape_layer_call_and_return_conditional_losses_356282d3�0
)�&
$�!
inputs���������

� "-�*
#� 
tensor_0����������
� �
-__inference_CDR2_reshape_layer_call_fn_356270Y3�0
)�&
$�!
inputs���������

� ""�
unknown�����������
E__inference_CDR3_conv_layer_call_and_return_conditional_losses_356194k'(3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
*__inference_CDR3_conv_layer_call_fn_356178`'(3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
H__inference_CDR3_flatten_layer_call_and_return_conditional_losses_356218d3�0
)�&
$�!
inputs���������
� "-�*
#� 
tensor_0����������
� �
-__inference_CDR3_flatten_layer_call_fn_356212Y3�0
)�&
$�!
inputs���������
� ""�
unknown�����������
K__inference_CDR3_maxpooling_layer_call_and_return_conditional_losses_356207�E�B
;�8
6�3
inputs'���������������������������
� "B�?
8�5
tensor_0'���������������������������
� �
0__inference_CDR3_maxpooling_layer_call_fn_356199�E�B
;�8
6�3
inputs'���������������������������
� "7�4
unknown'����������������������������
H__inference_CDR3_reshape_layer_call_and_return_conditional_losses_356235d3�0
)�&
$�!
inputs���������
� "-�*
#� 
tensor_0����������
� �
-__inference_CDR3_reshape_layer_call_fn_356223Y3�0
)�&
$�!
inputs���������
� ""�
unknown�����������
E__inference_HLA_dense_layer_call_and_return_conditional_losses_356355dfg0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
*__inference_HLA_dense_layer_call_fn_356344Yfg0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
G__inference_HLA_reshape_layer_call_and_return_conditional_losses_356316d3�0
)�&
$�!
inputs���������-
� "-�*
#� 
tensor_0����������
� �
,__inference_HLA_reshape_layer_call_fn_356304Y3�0
)�&
$�!
inputs���������-
� ""�
unknown�����������
E__inference_TCR_dense_layer_call_and_return_conditional_losses_356375dno0�-
&�#
!�
inputs����������

� ",�)
"�
tensor_0���������@
� �
*__inference_TCR_dense_layer_call_fn_356364Yno0�-
&�#
!�
inputs����������

� "!�
unknown���������@�
!__inference__wrapped_model_355658�'(fgno|}�������
���
���
!�
HLA���������-
"�
CDR3���������
"�
CDR3_len���������
"�
CDR1���������
"�
CDR2���������

$�!
CDR2.5���������
� "/�,
*
output �
output����������
I__inference_concatenate_1_layer_call_and_return_conditional_losses_356335����
���
���
"�
inputs_0���������
#� 
inputs_1����������
#� 
inputs_2����������
#� 
inputs_3����������
"�
inputs_4���������~
� "-�*
#� 
tensor_0����������

� �
.__inference_concatenate_1_layer_call_fn_356325����
���
���
"�
inputs_0���������
#� 
inputs_1����������
#� 
inputs_2����������
#� 
inputs_3����������
"�
inputs_4���������~
� ""�
unknown����������
�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_356388�Z�W
P�M
K�H
"�
inputs_0���������@
"�
inputs_1���������@
� "-�*
#� 
tensor_0����������
� �
.__inference_concatenate_2_layer_call_fn_356381�Z�W
P�M
K�H
"�
inputs_0���������@
"�
inputs_1���������@
� ""�
unknown�����������
G__inference_concatenate_layer_call_and_return_conditional_losses_356248�\�Y
R�O
M�J
#� 
inputs_0����������
#� 
inputs_1����������
� "-�*
#� 
tensor_0����������
� �
,__inference_concatenate_layer_call_fn_356241�\�Y
R�O
M�J
#� 
inputs_0����������
#� 
inputs_1����������
� ""�
unknown�����������
C__inference_dropout_layer_call_and_return_conditional_losses_356430c3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
C__inference_dropout_layer_call_and_return_conditional_losses_356435c3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
(__inference_dropout_layer_call_fn_356413X3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
(__inference_dropout_layer_call_fn_356418X3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@�
A__inference_model_layer_call_and_return_conditional_losses_355894�'(fgno|}�������
���
���
!�
HLA���������-
"�
CDR3���������
"�
CDR3_len���������
"�
CDR1���������
"�
CDR2���������

$�!
CDR2.5���������
p

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_355949�'(fgno|}�������
���
���
!�
HLA���������-
"�
CDR3���������
"�
CDR3_len���������
"�
CDR1���������
"�
CDR2���������

$�!
CDR2.5���������
p 

 
� ",�)
"�
tensor_0���������
� �
&__inference_model_layer_call_fn_355983�'(fgno|}�������
���
���
!�
HLA���������-
"�
CDR3���������
"�
CDR3_len���������
"�
CDR1���������
"�
CDR2���������

$�!
CDR2.5���������
p

 
� "!�
unknown����������
&__inference_model_layer_call_fn_356017�'(fgno|}�������
���
���
!�
HLA���������-
"�
CDR3���������
"�
CDR3_len���������
"�
CDR1���������
"�
CDR2���������

$�!
CDR2.5���������
p 

 
� "!�
unknown����������
B__inference_output_layer_call_and_return_conditional_losses_356475e��/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
'__inference_output_layer_call_fn_356464Z��/�,
%�"
 �
inputs���������
� "!�
unknown����������
G__inference_pair_dense1_layer_call_and_return_conditional_losses_356408d|}0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
,__inference_pair_dense1_layer_call_fn_356397Y|}0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
G__inference_pair_dense2_layer_call_and_return_conditional_losses_356455e��/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
,__inference_pair_dense2_layer_call_fn_356444Z��/�,
%�"
 �
inputs���������@
� "!�
unknown����������
$__inference_signature_wrapper_356169�'(fgno|}�������
� 
���
*
CDR1"�
cdr1���������
.
CDR2_5$�!
CDR2.5���������
*
CDR2"�
cdr2���������

.
CDR3_len"�
cdr3_len���������
*
CDR3"�
cdr3���������
(
HLA!�
hla���������-"/�,
*
output �
output���������