ĸ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
P
Shape

input"T
output"out_type"	
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58Ҕ
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
Adam/v/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/output/bias
u
&Adam/v/output/bias/Read/ReadVariableOpReadVariableOpAdam/v/output/bias*
_output_shapes
:*
dtype0
|
Adam/m/output/biasVarHandleOp*
_output_shapes
: *
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
: *
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
: *
dtype0*
shape
:*%
shared_nameAdam/m/output/kernel
}
(Adam/m/output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/output/kernel*
_output_shapes

:*
dtype0
~
Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_3/bias
w
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_3/bias
w
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/v/dense_3/kernel

)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes

:@*
dtype0
�
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/m/dense_3/kernel

)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes

:@*
dtype0
~
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/v/dense_2/kernel
�
)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/m/dense_2/kernel
�
)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes
:	�@*
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
@*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	�
@*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
@*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	�
@*
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
:@*
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/Conv_CDR3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/Conv_CDR3_1/bias

+Adam/v/Conv_CDR3_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/Conv_CDR3_1/bias*
_output_shapes
:*
dtype0
�
Adam/m/Conv_CDR3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/Conv_CDR3_1/bias

+Adam/m/Conv_CDR3_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/Conv_CDR3_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/Conv_CDR3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/v/Conv_CDR3_1/kernel
�
-Adam/v/Conv_CDR3_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Conv_CDR3_1/kernel*"
_output_shapes
:*
dtype0
�
Adam/m/Conv_CDR3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/m/Conv_CDR3_1/kernel
�
-Adam/m/Conv_CDR3_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Conv_CDR3_1/kernel*"
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	�@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�
@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�@*
dtype0
x
Conv_CDR3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameConv_CDR3_1/bias
q
$Conv_CDR3_1/bias/Read/ReadVariableOpReadVariableOpConv_CDR3_1/bias*
_output_shapes
:*
dtype0
�
Conv_CDR3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameConv_CDR3_1/kernel
}
&Conv_CDR3_1/kernel/Read/ReadVariableOpReadVariableOpConv_CDR3_1/kernel*"
_output_shapes
:*
dtype0
�
serving_default_input_1Placeholder*+
_output_shapes
:���������-*
dtype0* 
shape:���������-
�
serving_default_input_2Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
z
serving_default_input_3Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_input_4Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
serving_default_input_5Placeholder*+
_output_shapes
:���������
*
dtype0* 
shape:���������

�
serving_default_input_6Placeholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4serving_default_input_5serving_default_input_6Conv_CDR3_1/kernelConv_CDR3_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasoutput/kerneloutput/bias*
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
$__inference_signature_wrapper_366437

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
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
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
b\
VARIABLE_VALUEConv_CDR3_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEConv_CDR3_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

�trace_0* 
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
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
d^
VARIABLE_VALUEAdam/m/Conv_CDR3_1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/Conv_CDR3_1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/Conv_CDR3_1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/Conv_CDR3_1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/dense/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_3/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_3/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&Conv_CDR3_1/kernel/Read/ReadVariableOp$Conv_CDR3_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp-Adam/m/Conv_CDR3_1/kernel/Read/ReadVariableOp-Adam/v/Conv_CDR3_1/kernel/Read/ReadVariableOp+Adam/m/Conv_CDR3_1/bias/Read/ReadVariableOp+Adam/v/Conv_CDR3_1/bias/Read/ReadVariableOp'Adam/m/dense/kernel/Read/ReadVariableOp'Adam/v/dense/kernel/Read/ReadVariableOp%Adam/m/dense/bias/Read/ReadVariableOp%Adam/v/dense/bias/Read/ReadVariableOp)Adam/m/dense_1/kernel/Read/ReadVariableOp)Adam/v/dense_1/kernel/Read/ReadVariableOp'Adam/m/dense_1/bias/Read/ReadVariableOp'Adam/v/dense_1/bias/Read/ReadVariableOp)Adam/m/dense_2/kernel/Read/ReadVariableOp)Adam/v/dense_2/kernel/Read/ReadVariableOp'Adam/m/dense_2/bias/Read/ReadVariableOp'Adam/v/dense_2/bias/Read/ReadVariableOp)Adam/m/dense_3/kernel/Read/ReadVariableOp)Adam/v/dense_3/kernel/Read/ReadVariableOp'Adam/m/dense_3/bias/Read/ReadVariableOp'Adam/v/dense_3/bias/Read/ReadVariableOp(Adam/m/output/kernel/Read/ReadVariableOp(Adam/v/output/kernel/Read/ReadVariableOp&Adam/m/output/bias/Read/ReadVariableOp&Adam/v/output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOpConst*9
Tin2
02.	*
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
__inference__traced_save_367196
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv_CDR3_1/kernelConv_CDR3_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasoutput/kerneloutput/bias	iterationlearning_rateAdam/m/Conv_CDR3_1/kernelAdam/v/Conv_CDR3_1/kernelAdam/m/Conv_CDR3_1/biasAdam/v/Conv_CDR3_1/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/output/kernelAdam/v/output/kernelAdam/m/output/biasAdam/v/output/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives*8
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
"__inference__traced_restore_367338��
�

�
C__inference_dense_3_layer_call_and_return_conditional_losses_367016

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
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
s
G__inference_concatenate_layer_call_and_return_conditional_losses_366809
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
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1
�
q
G__inference_concatenate_layer_call_and_return_conditional_losses_365805

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
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�~
�	
A__inference_model_layer_call_and_return_conditional_losses_366730
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5M
7conv_cdr3_1_conv1d_expanddims_1_readvariableop_resource:9
+conv_cdr3_1_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@9
&dense_1_matmul_readvariableop_resource:	�
@5
'dense_1_biasadd_readvariableop_resource:@9
&dense_2_matmul_readvariableop_resource:	�@5
'dense_2_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity��"Conv_CDR3_1/BiasAdd/ReadVariableOp�.Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpl
!Conv_CDR3_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv_CDR3_1/Conv1D/ExpandDims
ExpandDimsinputs_1*Conv_CDR3_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
.Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7conv_cdr3_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0e
#Conv_CDR3_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv_CDR3_1/Conv1D/ExpandDims_1
ExpandDims6Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0,Conv_CDR3_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv_CDR3_1/Conv1DConv2D&Conv_CDR3_1/Conv1D/ExpandDims:output:0(Conv_CDR3_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv_CDR3_1/Conv1D/SqueezeSqueezeConv_CDR3_1/Conv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
"Conv_CDR3_1/BiasAdd/ReadVariableOpReadVariableOp+conv_cdr3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Conv_CDR3_1/BiasAddBiasAdd#Conv_CDR3_1/Conv1D/Squeeze:output:0*Conv_CDR3_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������l
Conv_CDR3_1/ReluReluConv_CDR3_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������b
 MaxPooling_CDR3_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
MaxPooling_CDR3_1/ExpandDims
ExpandDimsConv_CDR3_1/Relu:activations:0)MaxPooling_CDR3_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
MaxPooling_CDR3_1/MaxPoolMaxPool%MaxPooling_CDR3_1/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
MaxPooling_CDR3_1/SqueezeSqueeze"MaxPooling_CDR3_1/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
c
Flatten_CDR3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Flatten_CDR3/ReshapeReshape"MaxPooling_CDR3_1/Squeeze:output:0Flatten_CDR3/Const:output:0*
T0*(
_output_shapes
:����������G
reshape_4/ShapeShapeinputs_1*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
reshape_4/ReshapeReshapeinputs_1 reshape_4/Reshape/shape:output:0*
T0*(
_output_shapes
:����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2Flatten_CDR3/Reshape:output:0reshape_4/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������G
reshape_1/ShapeShapeinputs_3*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
reshape_1/ReshapeReshapeinputs_3 reshape_1/Reshape/shape:output:0*
T0*(
_output_shapes
:����������G
reshape_2/ShapeShapeinputs_4*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
reshape_2/ReshapeReshapeinputs_4 reshape_2/Reshape/shape:output:0*
T0*(
_output_shapes
:����������G
reshape_3/ShapeShapeinputs_5*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~�
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:z
reshape_3/ReshapeReshapeinputs_5 reshape_3/Reshape/shape:output:0*
T0*'
_output_shapes
:���������~[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2inputs_2concatenate/concat:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������
E
reshape/ShapeShapeinputs_0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:w
reshape/ReshapeReshapeinputs_0reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense/MatMulMatMulreshape/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�
@*
dtype0�
dense_1/MatMulMatMulconcatenate_1/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2dense/Relu:activations:0dense_1/Relu:activations:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_2/MatMulMatMulconcatenate_2/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout/dropout/MulMuldense_2/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@_
dropout/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�1c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_3/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:����������
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
output/MatMulMatMuldense_3/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^Conv_CDR3_1/BiasAdd/ReadVariableOp/^Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2H
"Conv_CDR3_1/BiasAdd/ReadVariableOp"Conv_CDR3_1/BiasAdd/ReadVariableOp2`
.Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp.Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:U Q
+
_output_shapes
:���������-
"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_3:UQ
+
_output_shapes
:���������

"
_user_specified_name
inputs_4:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_5
�	
�
.__inference_concatenate_1_layer_call_fn_366886
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
I__inference_concatenate_1_layer_call_and_return_conditional_losses_365859a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*u
_input_shapesd
b:���������:����������:����������:����������:���������~:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_2:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������~
"
_user_specified_name
inputs_4
�
�
,__inference_Conv_CDR3_1_layer_call_fn_366739

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
GPU 2J 8� *P
fKRI
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_365769s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�D
�
A__inference_model_layer_call_and_return_conditional_losses_366238

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5(
conv_cdr3_1_366196: 
conv_cdr3_1_366198:
dense_366210:	�@
dense_366212:@!
dense_1_366215:	�
@
dense_1_366217:@!
dense_2_366221:	�@
dense_2_366223:@ 
dense_3_366227:@
dense_3_366229:
output_366232:
output_366234:
identity��#Conv_CDR3_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dropout/StatefulPartitionedCall�output/StatefulPartitionedCall�
#Conv_CDR3_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv_cdr3_1_366196conv_cdr3_1_366198*
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
GPU 2J 8� *P
fKRI
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_365769�
!MaxPooling_CDR3_1/PartitionedCallPartitionedCall,Conv_CDR3_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *V
fQRO
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_365733�
Flatten_CDR3/PartitionedCallPartitionedCall*MaxPooling_CDR3_1/PartitionedCall:output:0*
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
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_365782�
reshape_4/PartitionedCallPartitionedCallinputs_1*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_365796�
concatenate/PartitionedCallPartitionedCall%Flatten_CDR3/PartitionedCall:output:0"reshape_4/PartitionedCall:output:0*
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
G__inference_concatenate_layer_call_and_return_conditional_losses_365805�
reshape_1/PartitionedCallPartitionedCallinputs_3*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365819�
reshape_2/PartitionedCallPartitionedCallinputs_4*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_365833�
reshape_3/PartitionedCallPartitionedCallinputs_5*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_365847�
concatenate_1/PartitionedCallPartitionedCallinputs_2$concatenate/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
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
I__inference_concatenate_1_layer_call_and_return_conditional_losses_365859�
reshape/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_365873�
dense/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dense_366210dense_366212*
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
GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_365886�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_1_366215dense_1_366217*
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
GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365903�
concatenate_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_365916�
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_366221dense_2_366223*
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
GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_365929�
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_366044�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_3_366227dense_3_366229*
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
GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_365953�
output/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0output_366232output_366234*
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
B__inference_output_layer_call_and_return_conditional_losses_365970v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^Conv_CDR3_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2J
#Conv_CDR3_1/StatefulPartitionedCall#Conv_CDR3_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:���������-
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������

 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_366505
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
A__inference_model_layer_call_and_return_conditional_losses_366238o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������-
"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_3:UQ
+
_output_shapes
:���������

"
_user_specified_name
inputs_4:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_5
�
F
*__inference_reshape_1_layer_call_fn_366814

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
GPU 2J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365819a
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
B__inference_output_layer_call_and_return_conditional_losses_367036

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
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_output_layer_call_fn_367025

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
B__inference_output_layer_call_and_return_conditional_losses_365970o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
F
*__inference_reshape_3_layer_call_fn_366848

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
GPU 2J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_365847`
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
�

�
B__inference_output_layer_call_and_return_conditional_losses_365970

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
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�

!__inference__wrapped_model_365721
input_1
input_2
input_3
input_4
input_5
input_6S
=model_conv_cdr3_1_conv1d_expanddims_1_readvariableop_resource:?
1model_conv_cdr3_1_biasadd_readvariableop_resource:=
*model_dense_matmul_readvariableop_resource:	�@9
+model_dense_biasadd_readvariableop_resource:@?
,model_dense_1_matmul_readvariableop_resource:	�
@;
-model_dense_1_biasadd_readvariableop_resource:@?
,model_dense_2_matmul_readvariableop_resource:	�@;
-model_dense_2_biasadd_readvariableop_resource:@>
,model_dense_3_matmul_readvariableop_resource:@;
-model_dense_3_biasadd_readvariableop_resource:=
+model_output_matmul_readvariableop_resource::
,model_output_biasadd_readvariableop_resource:
identity��(model/Conv_CDR3_1/BiasAdd/ReadVariableOp�4model/Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�$model/dense_3/BiasAdd/ReadVariableOp�#model/dense_3/MatMul/ReadVariableOp�#model/output/BiasAdd/ReadVariableOp�"model/output/MatMul/ReadVariableOpr
'model/Conv_CDR3_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
#model/Conv_CDR3_1/Conv1D/ExpandDims
ExpandDimsinput_20model/Conv_CDR3_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
4model/Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=model_conv_cdr3_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0k
)model/Conv_CDR3_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
%model/Conv_CDR3_1/Conv1D/ExpandDims_1
ExpandDims<model/Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp:value:02model/Conv_CDR3_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
model/Conv_CDR3_1/Conv1DConv2D,model/Conv_CDR3_1/Conv1D/ExpandDims:output:0.model/Conv_CDR3_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
 model/Conv_CDR3_1/Conv1D/SqueezeSqueeze!model/Conv_CDR3_1/Conv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
(model/Conv_CDR3_1/BiasAdd/ReadVariableOpReadVariableOp1model_conv_cdr3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/Conv_CDR3_1/BiasAddBiasAdd)model/Conv_CDR3_1/Conv1D/Squeeze:output:00model/Conv_CDR3_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������x
model/Conv_CDR3_1/ReluRelu"model/Conv_CDR3_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������h
&model/MaxPooling_CDR3_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
"model/MaxPooling_CDR3_1/ExpandDims
ExpandDims$model/Conv_CDR3_1/Relu:activations:0/model/MaxPooling_CDR3_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
model/MaxPooling_CDR3_1/MaxPoolMaxPool+model/MaxPooling_CDR3_1/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
model/MaxPooling_CDR3_1/SqueezeSqueeze(model/MaxPooling_CDR3_1/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
i
model/Flatten_CDR3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
model/Flatten_CDR3/ReshapeReshape(model/MaxPooling_CDR3_1/Squeeze:output:0!model/Flatten_CDR3/Const:output:0*
T0*(
_output_shapes
:����������L
model/reshape_4/ShapeShapeinput_2*
T0*
_output_shapes
:m
#model/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape_4/strided_sliceStridedSlicemodel/reshape_4/Shape:output:0,model/reshape_4/strided_slice/stack:output:0.model/reshape_4/strided_slice/stack_1:output:0.model/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
model/reshape_4/Reshape/shapePack&model/reshape_4/strided_slice:output:0(model/reshape_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/reshape_4/ReshapeReshapeinput_2&model/reshape_4/Reshape/shape:output:0*
T0*(
_output_shapes
:����������_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2#model/Flatten_CDR3/Reshape:output:0 model/reshape_4/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������L
model/reshape_1/ShapeShapeinput_4*
T0*
_output_shapes
:m
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/reshape_1/ReshapeReshapeinput_4&model/reshape_1/Reshape/shape:output:0*
T0*(
_output_shapes
:����������L
model/reshape_2/ShapeShapeinput_5*
T0*
_output_shapes
:m
#model/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape_2/strided_sliceStridedSlicemodel/reshape_2/Shape:output:0,model/reshape_2/strided_slice/stack:output:0.model/reshape_2/strided_slice/stack_1:output:0.model/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
model/reshape_2/Reshape/shapePack&model/reshape_2/strided_slice:output:0(model/reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/reshape_2/ReshapeReshapeinput_5&model/reshape_2/Reshape/shape:output:0*
T0*(
_output_shapes
:����������L
model/reshape_3/ShapeShapeinput_6*
T0*
_output_shapes
:m
#model/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape_3/strided_sliceStridedSlicemodel/reshape_3/Shape:output:0,model/reshape_3/strided_slice/stack:output:0.model/reshape_3/strided_slice/stack_1:output:0.model/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~�
model/reshape_3/Reshape/shapePack&model/reshape_3/strided_slice:output:0(model/reshape_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/reshape_3/ReshapeReshapeinput_6&model/reshape_3/Reshape/shape:output:0*
T0*'
_output_shapes
:���������~a
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate_1/concatConcatV2input_3!model/concatenate/concat:output:0 model/reshape_1/Reshape:output:0 model/reshape_2/Reshape:output:0 model/reshape_3/Reshape:output:0(model/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������
J
model/reshape/ShapeShapeinput_1*
T0*
_output_shapes
:k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:�
model/reshape/ReshapeReshapeinput_1$model/reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense/MatMulMatMulmodel/reshape/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@h
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�
@*
dtype0�
model/dense_1/MatMulMatMul#model/concatenate_1/concat:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@a
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate_2/concatConcatV2model/dense/Relu:activations:0 model/dense_1/Relu:activations:0(model/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/dense_2/MatMulMatMul#model/concatenate_2/concat:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@v
model/dropout/IdentityIdentity model/dense_2/Relu:activations:0*
T0*'
_output_shapes
:���������@�
#model/dense_3/MatMul/ReadVariableOpReadVariableOp,model_dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/dense_3/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_3/BiasAddBiasAddmodel/dense_3/MatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
model/dense_3/ReluRelumodel/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:����������
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model/output/MatMulMatMul model/dense_3/Relu:activations:0*model/output/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp)^model/Conv_CDR3_1/BiasAdd/ReadVariableOp5^model/Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp$^model/dense_3/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2T
(model/Conv_CDR3_1/BiasAdd/ReadVariableOp(model/Conv_CDR3_1/BiasAdd/ReadVariableOp2l
4model/Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp4model/Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2J
#model/dense_3/MatMul/ReadVariableOp#model/dense_3/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:T P
+
_output_shapes
:���������-
!
_user_specified_name	input_1:TP
+
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:TP
+
_output_shapes
:���������
!
_user_specified_name	input_4:TP
+
_output_shapes
:���������

!
_user_specified_name	input_5:TP
+
_output_shapes
:���������
!
_user_specified_name	input_6
�	
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_366826

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
�
&__inference_model_layer_call_fn_366471
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
A__inference_model_layer_call_and_return_conditional_losses_365977o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������-
"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_3:UQ
+
_output_shapes
:���������

"
_user_specified_name
inputs_4:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_5
�	
a
E__inference_reshape_4_layer_call_and_return_conditional_losses_365796

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
�v
�	
A__inference_model_layer_call_and_return_conditional_losses_366614
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5M
7conv_cdr3_1_conv1d_expanddims_1_readvariableop_resource:9
+conv_cdr3_1_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@9
&dense_1_matmul_readvariableop_resource:	�
@5
'dense_1_biasadd_readvariableop_resource:@9
&dense_2_matmul_readvariableop_resource:	�@5
'dense_2_biasadd_readvariableop_resource:@8
&dense_3_matmul_readvariableop_resource:@5
'dense_3_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity��"Conv_CDR3_1/BiasAdd/ReadVariableOp�.Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOpl
!Conv_CDR3_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv_CDR3_1/Conv1D/ExpandDims
ExpandDimsinputs_1*Conv_CDR3_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
.Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7conv_cdr3_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype0e
#Conv_CDR3_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv_CDR3_1/Conv1D/ExpandDims_1
ExpandDims6Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0,Conv_CDR3_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:�
Conv_CDR3_1/Conv1DConv2D&Conv_CDR3_1/Conv1D/ExpandDims:output:0(Conv_CDR3_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
Conv_CDR3_1/Conv1D/SqueezeSqueezeConv_CDR3_1/Conv1D:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
"Conv_CDR3_1/BiasAdd/ReadVariableOpReadVariableOp+conv_cdr3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Conv_CDR3_1/BiasAddBiasAdd#Conv_CDR3_1/Conv1D/Squeeze:output:0*Conv_CDR3_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������l
Conv_CDR3_1/ReluReluConv_CDR3_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������b
 MaxPooling_CDR3_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
MaxPooling_CDR3_1/ExpandDims
ExpandDimsConv_CDR3_1/Relu:activations:0)MaxPooling_CDR3_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
MaxPooling_CDR3_1/MaxPoolMaxPool%MaxPooling_CDR3_1/ExpandDims:output:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
MaxPooling_CDR3_1/SqueezeSqueeze"MaxPooling_CDR3_1/MaxPool:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims
c
Flatten_CDR3/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
Flatten_CDR3/ReshapeReshape"MaxPooling_CDR3_1/Squeeze:output:0Flatten_CDR3/Const:output:0*
T0*(
_output_shapes
:����������G
reshape_4/ShapeShapeinputs_1*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
reshape_4/ReshapeReshapeinputs_1 reshape_4/Reshape/shape:output:0*
T0*(
_output_shapes
:����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2Flatten_CDR3/Reshape:output:0reshape_4/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������G
reshape_1/ShapeShapeinputs_3*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
reshape_1/ReshapeReshapeinputs_3 reshape_1/Reshape/shape:output:0*
T0*(
_output_shapes
:����������G
reshape_2/ShapeShapeinputs_4*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:{
reshape_2/ReshapeReshapeinputs_4 reshape_2/Reshape/shape:output:0*
T0*(
_output_shapes
:����������G
reshape_3/ShapeShapeinputs_5*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~�
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:z
reshape_3/ReshapeReshapeinputs_5 reshape_3/Reshape/shape:output:0*
T0*'
_output_shapes
:���������~[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2inputs_2concatenate/concat:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������
E
reshape/ShapeShapeinputs_0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :��
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:w
reshape/ReshapeReshapeinputs_0reshape/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense/MatMulMatMulreshape/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�
@*
dtype0�
dense_1/MatMulMatMulconcatenate_1/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_2/concatConcatV2dense/Relu:activations:0dense_1/Relu:activations:0"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
dense_2/MatMulMatMulconcatenate_2/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@j
dropout/IdentityIdentitydense_2/Relu:activations:0*
T0*'
_output_shapes
:���������@�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_3/MatMulMatMuldropout/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:����������
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
output/MatMulMatMuldense_3/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityoutput/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp#^Conv_CDR3_1/BiasAdd/ReadVariableOp/^Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2H
"Conv_CDR3_1/BiasAdd/ReadVariableOp"Conv_CDR3_1/BiasAdd/ReadVariableOp2`
.Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp.Conv_CDR3_1/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:U Q
+
_output_shapes
:���������-
"
_user_specified_name
inputs_0:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_3:UQ
+
_output_shapes
:���������

"
_user_specified_name
inputs_4:UQ
+
_output_shapes
:���������
"
_user_specified_name
inputs_5
�D
�
A__inference_model_layer_call_and_return_conditional_losses_366399
input_1
input_2
input_3
input_4
input_5
input_6(
conv_cdr3_1_366357: 
conv_cdr3_1_366359:
dense_366371:	�@
dense_366373:@!
dense_1_366376:	�
@
dense_1_366378:@!
dense_2_366382:	�@
dense_2_366384:@ 
dense_3_366388:@
dense_3_366390:
output_366393:
output_366395:
identity��#Conv_CDR3_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dropout/StatefulPartitionedCall�output/StatefulPartitionedCall�
#Conv_CDR3_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv_cdr3_1_366357conv_cdr3_1_366359*
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
GPU 2J 8� *P
fKRI
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_365769�
!MaxPooling_CDR3_1/PartitionedCallPartitionedCall,Conv_CDR3_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *V
fQRO
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_365733�
Flatten_CDR3/PartitionedCallPartitionedCall*MaxPooling_CDR3_1/PartitionedCall:output:0*
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
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_365782�
reshape_4/PartitionedCallPartitionedCallinput_2*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_365796�
concatenate/PartitionedCallPartitionedCall%Flatten_CDR3/PartitionedCall:output:0"reshape_4/PartitionedCall:output:0*
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
G__inference_concatenate_layer_call_and_return_conditional_losses_365805�
reshape_1/PartitionedCallPartitionedCallinput_4*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365819�
reshape_2/PartitionedCallPartitionedCallinput_5*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_365833�
reshape_3/PartitionedCallPartitionedCallinput_6*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_365847�
concatenate_1/PartitionedCallPartitionedCallinput_3$concatenate/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
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
I__inference_concatenate_1_layer_call_and_return_conditional_losses_365859�
reshape/PartitionedCallPartitionedCallinput_1*
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
GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_365873�
dense/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dense_366371dense_366373*
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
GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_365886�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_1_366376dense_1_366378*
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
GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365903�
concatenate_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_365916�
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_366382dense_2_366384*
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
GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_365929�
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_366044�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_3_366388dense_3_366390*
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
GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_365953�
output/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0output_366393output_366395*
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
B__inference_output_layer_call_and_return_conditional_losses_365970v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^Conv_CDR3_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2J
#Conv_CDR3_1/StatefulPartitionedCall#Conv_CDR3_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
+
_output_shapes
:���������-
!
_user_specified_name	input_1:TP
+
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:TP
+
_output_shapes
:���������
!
_user_specified_name	input_4:TP
+
_output_shapes
:���������

!
_user_specified_name	input_5:TP
+
_output_shapes
:���������
!
_user_specified_name	input_6
�U
�
__inference__traced_save_367196
file_prefix1
-savev2_conv_cdr3_1_kernel_read_readvariableop/
+savev2_conv_cdr3_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop8
4savev2_adam_m_conv_cdr3_1_kernel_read_readvariableop8
4savev2_adam_v_conv_cdr3_1_kernel_read_readvariableop6
2savev2_adam_m_conv_cdr3_1_bias_read_readvariableop6
2savev2_adam_v_conv_cdr3_1_bias_read_readvariableop2
.savev2_adam_m_dense_kernel_read_readvariableop2
.savev2_adam_v_dense_kernel_read_readvariableop0
,savev2_adam_m_dense_bias_read_readvariableop0
,savev2_adam_v_dense_bias_read_readvariableop4
0savev2_adam_m_dense_1_kernel_read_readvariableop4
0savev2_adam_v_dense_1_kernel_read_readvariableop2
.savev2_adam_m_dense_1_bias_read_readvariableop2
.savev2_adam_v_dense_1_bias_read_readvariableop4
0savev2_adam_m_dense_2_kernel_read_readvariableop4
0savev2_adam_v_dense_2_kernel_read_readvariableop2
.savev2_adam_m_dense_2_bias_read_readvariableop2
.savev2_adam_v_dense_2_bias_read_readvariableop4
0savev2_adam_m_dense_3_kernel_read_readvariableop4
0savev2_adam_v_dense_3_kernel_read_readvariableop2
.savev2_adam_m_dense_3_bias_read_readvariableop2
.savev2_adam_v_dense_3_bias_read_readvariableop3
/savev2_adam_m_output_kernel_read_readvariableop3
/savev2_adam_v_output_kernel_read_readvariableop1
-savev2_adam_m_output_bias_read_readvariableop1
-savev2_adam_v_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_conv_cdr3_1_kernel_read_readvariableop+savev2_conv_cdr3_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop4savev2_adam_m_conv_cdr3_1_kernel_read_readvariableop4savev2_adam_v_conv_cdr3_1_kernel_read_readvariableop2savev2_adam_m_conv_cdr3_1_bias_read_readvariableop2savev2_adam_v_conv_cdr3_1_bias_read_readvariableop.savev2_adam_m_dense_kernel_read_readvariableop.savev2_adam_v_dense_kernel_read_readvariableop,savev2_adam_m_dense_bias_read_readvariableop,savev2_adam_v_dense_bias_read_readvariableop0savev2_adam_m_dense_1_kernel_read_readvariableop0savev2_adam_v_dense_1_kernel_read_readvariableop.savev2_adam_m_dense_1_bias_read_readvariableop.savev2_adam_v_dense_1_bias_read_readvariableop0savev2_adam_m_dense_2_kernel_read_readvariableop0savev2_adam_v_dense_2_kernel_read_readvariableop.savev2_adam_m_dense_2_bias_read_readvariableop.savev2_adam_v_dense_2_bias_read_readvariableop0savev2_adam_m_dense_3_kernel_read_readvariableop0savev2_adam_v_dense_3_kernel_read_readvariableop.savev2_adam_m_dense_3_bias_read_readvariableop.savev2_adam_v_dense_3_bias_read_readvariableop/savev2_adam_m_output_kernel_read_readvariableop/savev2_adam_v_output_kernel_read_readvariableop-savev2_adam_m_output_bias_read_readvariableop-savev2_adam_v_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableopsavev2_const"/device:CPU:0*&
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
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::	�@:@:	�
@:@:	�@:@:@:::: : :::::	�@:	�@:@:@:	�
@:	�
@:@:@:	�@:	�@:@:@:@:@::::::: : :�:�:�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�@: 

_output_shapes
:@:%!

_output_shapes
:	�
@: 

_output_shapes
:@:%!

_output_shapes
:	�@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	�@:%!

_output_shapes
:	�@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	�
@:%!

_output_shapes
:	�
@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	�@:%!

_output_shapes
:	�@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@:$  

_output_shapes

:@: !

_output_shapes
:: "

_output_shapes
::$# 

_output_shapes

::$$ 

_output_shapes

:: %

_output_shapes
:: &

_output_shapes
::'

_output_shapes
: :(

_output_shapes
: :!)

_output_shapes	
:�:!*

_output_shapes	
:�:!+

_output_shapes	
:�:!,

_output_shapes	
:�:-

_output_shapes
: 
�	
a
E__inference_reshape_2_layer_call_and_return_conditional_losses_366843

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
(__inference_dense_1_layer_call_fn_366925

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
GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365903o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_366996

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
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�1[
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
�
�
&__inference_model_layer_call_fn_366004
input_1
input_2
input_3
input_4
input_5
input_6
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
A__inference_model_layer_call_and_return_conditional_losses_365977o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������-
!
_user_specified_name	input_1:TP
+
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:TP
+
_output_shapes
:���������
!
_user_specified_name	input_4:TP
+
_output_shapes
:���������

!
_user_specified_name	input_5:TP
+
_output_shapes
:���������
!
_user_specified_name	input_6
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_366969

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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
u
I__inference_concatenate_2_layer_call_and_return_conditional_losses_366949
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
&:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs_1
�
�
(__inference_dense_3_layer_call_fn_367005

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
GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_365953o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_dense_layer_call_fn_366905

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
GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_365886o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_reshape_4_layer_call_fn_366784

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
GPU 2J 8� *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_365796a
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
�	
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_365847

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
�
N
2__inference_MaxPooling_CDR3_1_layer_call_fn_366760

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
GPU 2J 8� *V
fQRO
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_365733v
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
�	
a
E__inference_reshape_4_layer_call_and_return_conditional_losses_366796

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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

b
C__inference_dropout_layer_call_and_return_conditional_losses_366044

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
:���������@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed�1[
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
�
�
$__inference_signature_wrapper_366437
input_1
input_2
input_3
input_4
input_5
input_6
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
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_365721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������-
!
_user_specified_name	input_1:TP
+
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:TP
+
_output_shapes
:���������
!
_user_specified_name	input_4:TP
+
_output_shapes
:���������

!
_user_specified_name	input_5:TP
+
_output_shapes
:���������
!
_user_specified_name	input_6
�
d
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_366779

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
�
�
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_366755

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
:����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
D
(__inference_dropout_layer_call_fn_366974

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
C__inference_dropout_layer_call_and_return_conditional_losses_365940`
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
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_366984

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
�

�
A__inference_dense_layer_call_and_return_conditional_losses_365886

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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�C
�
A__inference_model_layer_call_and_return_conditional_losses_366349
input_1
input_2
input_3
input_4
input_5
input_6(
conv_cdr3_1_366307: 
conv_cdr3_1_366309:
dense_366321:	�@
dense_366323:@!
dense_1_366326:	�
@
dense_1_366328:@!
dense_2_366332:	�@
dense_2_366334:@ 
dense_3_366338:@
dense_3_366340:
output_366343:
output_366345:
identity��#Conv_CDR3_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�output/StatefulPartitionedCall�
#Conv_CDR3_1/StatefulPartitionedCallStatefulPartitionedCallinput_2conv_cdr3_1_366307conv_cdr3_1_366309*
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
GPU 2J 8� *P
fKRI
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_365769�
!MaxPooling_CDR3_1/PartitionedCallPartitionedCall,Conv_CDR3_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *V
fQRO
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_365733�
Flatten_CDR3/PartitionedCallPartitionedCall*MaxPooling_CDR3_1/PartitionedCall:output:0*
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
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_365782�
reshape_4/PartitionedCallPartitionedCallinput_2*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_365796�
concatenate/PartitionedCallPartitionedCall%Flatten_CDR3/PartitionedCall:output:0"reshape_4/PartitionedCall:output:0*
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
G__inference_concatenate_layer_call_and_return_conditional_losses_365805�
reshape_1/PartitionedCallPartitionedCallinput_4*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365819�
reshape_2/PartitionedCallPartitionedCallinput_5*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_365833�
reshape_3/PartitionedCallPartitionedCallinput_6*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_365847�
concatenate_1/PartitionedCallPartitionedCallinput_3$concatenate/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
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
I__inference_concatenate_1_layer_call_and_return_conditional_losses_365859�
reshape/PartitionedCallPartitionedCallinput_1*
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
GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_365873�
dense/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dense_366321dense_366323*
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
GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_365886�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_1_366326dense_1_366328*
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
GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365903�
concatenate_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_365916�
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_366332dense_2_366334*
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
GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_365929�
dropout/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_365940�
dense_3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_3_366338dense_3_366340*
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
GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_365953�
output/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0output_366343output_366345*
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
B__inference_output_layer_call_and_return_conditional_losses_365970v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^Conv_CDR3_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2J
#Conv_CDR3_1/StatefulPartitionedCall#Conv_CDR3_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:T P
+
_output_shapes
:���������-
!
_user_specified_name	input_1:TP
+
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:TP
+
_output_shapes
:���������
!
_user_specified_name	input_4:TP
+
_output_shapes
:���������

!
_user_specified_name	input_5:TP
+
_output_shapes
:���������
!
_user_specified_name	input_6
�
D
(__inference_reshape_layer_call_fn_366865

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
GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_365873a
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
�
s
I__inference_concatenate_2_layer_call_and_return_conditional_losses_365916

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
&:���������@:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_3_layer_call_and_return_conditional_losses_365953

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
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_365929

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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_365940

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
X
,__inference_concatenate_layer_call_fn_366802
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
G__inference_concatenate_layer_call_and_return_conditional_losses_365805a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1
�

�
A__inference_dense_layer_call_and_return_conditional_losses_366916

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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_366768

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
a
E__inference_reshape_2_layer_call_and_return_conditional_losses_365833

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
(__inference_dense_2_layer_call_fn_366958

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
GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_365929o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_365819

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
�
Z
.__inference_concatenate_2_layer_call_fn_366942
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_365916a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������@:���������@:Q M
'
_output_shapes
:���������@
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:���������@
"
_user_specified_name
inputs_1
�	
_
C__inference_reshape_layer_call_and_return_conditional_losses_366877

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
�
i
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_365733

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
�
�
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_365769

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
:����������
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_367338
file_prefix9
#assignvariableop_conv_cdr3_1_kernel:1
#assignvariableop_1_conv_cdr3_1_bias:2
assignvariableop_2_dense_kernel:	�@+
assignvariableop_3_dense_bias:@4
!assignvariableop_4_dense_1_kernel:	�
@-
assignvariableop_5_dense_1_bias:@4
!assignvariableop_6_dense_2_kernel:	�@-
assignvariableop_7_dense_2_bias:@3
!assignvariableop_8_dense_3_kernel:@-
assignvariableop_9_dense_3_bias:3
!assignvariableop_10_output_kernel:-
assignvariableop_11_output_bias:'
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: C
-assignvariableop_14_adam_m_conv_cdr3_1_kernel:C
-assignvariableop_15_adam_v_conv_cdr3_1_kernel:9
+assignvariableop_16_adam_m_conv_cdr3_1_bias:9
+assignvariableop_17_adam_v_conv_cdr3_1_bias::
'assignvariableop_18_adam_m_dense_kernel:	�@:
'assignvariableop_19_adam_v_dense_kernel:	�@3
%assignvariableop_20_adam_m_dense_bias:@3
%assignvariableop_21_adam_v_dense_bias:@<
)assignvariableop_22_adam_m_dense_1_kernel:	�
@<
)assignvariableop_23_adam_v_dense_1_kernel:	�
@5
'assignvariableop_24_adam_m_dense_1_bias:@5
'assignvariableop_25_adam_v_dense_1_bias:@<
)assignvariableop_26_adam_m_dense_2_kernel:	�@<
)assignvariableop_27_adam_v_dense_2_kernel:	�@5
'assignvariableop_28_adam_m_dense_2_bias:@5
'assignvariableop_29_adam_v_dense_2_bias:@;
)assignvariableop_30_adam_m_dense_3_kernel:@;
)assignvariableop_31_adam_v_dense_3_kernel:@5
'assignvariableop_32_adam_m_dense_3_bias:5
'assignvariableop_33_adam_v_dense_3_bias::
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
AssignVariableOpAssignVariableOp#assignvariableop_conv_cdr3_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_conv_cdr3_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_3_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_3_biasIdentity_9:output:0"/device:CPU:0*&
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
AssignVariableOp_14AssignVariableOp-assignvariableop_14_adam_m_conv_cdr3_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp-assignvariableop_15_adam_v_conv_cdr3_1_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp+assignvariableop_16_adam_m_conv_cdr3_1_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_v_conv_cdr3_1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_m_dense_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_v_dense_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_m_dense_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_v_dense_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_dense_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_dense_1_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_m_dense_1_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_v_dense_1_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_m_dense_2_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_v_dense_2_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_m_dense_2_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_v_dense_2_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_m_dense_3_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_v_dense_3_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_m_dense_3_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_v_dense_3_biasIdentity_33:output:0"/device:CPU:0*&
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
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_45Identity_45:output:0*m
_input_shapes\
Z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
I__inference_concatenate_1_layer_call_and_return_conditional_losses_366896
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
b:���������:����������:����������:����������:���������~:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_1:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_2:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������~
"
_user_specified_name
inputs_4
�	
a
E__inference_reshape_3_layer_call_and_return_conditional_losses_366860

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
�
�
&__inference_model_layer_call_fn_366299
input_1
input_2
input_3
input_4
input_5
input_6
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
A__inference_model_layer_call_and_return_conditional_losses_366238o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������-
!
_user_specified_name	input_1:TP
+
_output_shapes
:���������
!
_user_specified_name	input_2:PL
'
_output_shapes
:���������
!
_user_specified_name	input_3:TP
+
_output_shapes
:���������
!
_user_specified_name	input_4:TP
+
_output_shapes
:���������

!
_user_specified_name	input_5:TP
+
_output_shapes
:���������
!
_user_specified_name	input_6
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_365903

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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�
F
*__inference_reshape_2_layer_call_fn_366831

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
GPU 2J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_365833a
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
�
a
(__inference_dropout_layer_call_fn_366979

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
C__inference_dropout_layer_call_and_return_conditional_losses_366044o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
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
�
d
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_365782

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
�C
�
A__inference_model_layer_call_and_return_conditional_losses_365977

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5(
conv_cdr3_1_365770: 
conv_cdr3_1_365772:
dense_365887:	�@
dense_365889:@!
dense_1_365904:	�
@
dense_1_365906:@!
dense_2_365930:	�@
dense_2_365932:@ 
dense_3_365954:@
dense_3_365956:
output_365971:
output_365973:
identity��#Conv_CDR3_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�output/StatefulPartitionedCall�
#Conv_CDR3_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv_cdr3_1_365770conv_cdr3_1_365772*
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
GPU 2J 8� *P
fKRI
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_365769�
!MaxPooling_CDR3_1/PartitionedCallPartitionedCall,Conv_CDR3_1/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *V
fQRO
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_365733�
Flatten_CDR3/PartitionedCallPartitionedCall*MaxPooling_CDR3_1/PartitionedCall:output:0*
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
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_365782�
reshape_4/PartitionedCallPartitionedCallinputs_1*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_4_layer_call_and_return_conditional_losses_365796�
concatenate/PartitionedCallPartitionedCall%Flatten_CDR3/PartitionedCall:output:0"reshape_4/PartitionedCall:output:0*
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
G__inference_concatenate_layer_call_and_return_conditional_losses_365805�
reshape_1/PartitionedCallPartitionedCallinputs_3*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_365819�
reshape_2/PartitionedCallPartitionedCallinputs_4*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_2_layer_call_and_return_conditional_losses_365833�
reshape_3/PartitionedCallPartitionedCallinputs_5*
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
GPU 2J 8� *N
fIRG
E__inference_reshape_3_layer_call_and_return_conditional_losses_365847�
concatenate_1/PartitionedCallPartitionedCallinputs_2$concatenate/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
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
I__inference_concatenate_1_layer_call_and_return_conditional_losses_365859�
reshape/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_365873�
dense/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dense_365887dense_365889*
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
GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_365886�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_1_365904dense_1_365906*
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
GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_365903�
concatenate_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0*
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_365916�
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_365930dense_2_365932*
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
GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_365929�
dropout/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
C__inference_dropout_layer_call_and_return_conditional_losses_365940�
dense_3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_3_365954dense_3_365956*
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
GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_365953�
output/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0output_365971output_365973*
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
B__inference_output_layer_call_and_return_conditional_losses_365970v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^Conv_CDR3_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������-:���������:���������:���������:���������
:���������: : : : : : : : : : : : 2J
#Conv_CDR3_1/StatefulPartitionedCall#Conv_CDR3_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:S O
+
_output_shapes
:���������-
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������

 
_user_specified_nameinputs:SO
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
_
C__inference_reshape_layer_call_and_return_conditional_losses_365873

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
�
I
-__inference_Flatten_CDR3_layer_call_fn_366773

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
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_365782a
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

�
C__inference_dense_1_layer_call_and_return_conditional_losses_366936

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
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�
�
I__inference_concatenate_1_layer_call_and_return_conditional_losses_365859

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
b:���������:����������:����������:����������:���������~:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������~
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������-
?
input_24
serving_default_input_2:0���������
;
input_30
serving_default_input_3:0���������
?
input_44
serving_default_input_4:0���������
?
input_54
serving_default_input_5:0���������

?
input_64
serving_default_input_6:0���������:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
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
�
�trace_0
�trace_1
�trace_2
�trace_32�
&__inference_model_layer_call_fn_366004
&__inference_model_layer_call_fn_366471
&__inference_model_layer_call_fn_366505
&__inference_model_layer_call_fn_366299�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
A__inference_model_layer_call_and_return_conditional_losses_366614
A__inference_model_layer_call_and_return_conditional_losses_366730
A__inference_model_layer_call_and_return_conditional_losses_366349
A__inference_model_layer_call_and_return_conditional_losses_366399�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
!__inference__wrapped_model_365721input_1input_2input_3input_4input_5input_6"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
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
,__inference_Conv_CDR3_1_layer_call_fn_366739�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_366755�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
(:&2Conv_CDR3_1/kernel
:2Conv_CDR3_1/bias
�2��
���
FullArgSpec'
args�
jself
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
annotations� *
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
2__inference_MaxPooling_CDR3_1_layer_call_fn_366760�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_366768�
���
FullArgSpec
args�
jself
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
annotations� *
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
-__inference_Flatten_CDR3_layer_call_fn_366773�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_366779�
���
FullArgSpec
args�
jself
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
annotations� *
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
*__inference_reshape_4_layer_call_fn_366784�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_reshape_4_layer_call_and_return_conditional_losses_366796�
���
FullArgSpec
args�
jself
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
annotations� *
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
,__inference_concatenate_layer_call_fn_366802�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
G__inference_concatenate_layer_call_and_return_conditional_losses_366809�
���
FullArgSpec
args�
jself
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
annotations� *
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
*__inference_reshape_1_layer_call_fn_366814�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_reshape_1_layer_call_and_return_conditional_losses_366826�
���
FullArgSpec
args�
jself
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
annotations� *
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
*__inference_reshape_2_layer_call_fn_366831�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_reshape_2_layer_call_and_return_conditional_losses_366843�
���
FullArgSpec
args�
jself
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
annotations� *
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
*__inference_reshape_3_layer_call_fn_366848�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_reshape_3_layer_call_and_return_conditional_losses_366860�
���
FullArgSpec
args�
jself
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
annotations� *
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
(__inference_reshape_layer_call_fn_366865�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_reshape_layer_call_and_return_conditional_losses_366877�
���
FullArgSpec
args�
jself
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
annotations� *
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
.__inference_concatenate_1_layer_call_fn_366886�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_1_layer_call_and_return_conditional_losses_366896�
���
FullArgSpec
args�
jself
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
annotations� *
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
&__inference_dense_layer_call_fn_366905�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_366916�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
:	�@2dense/kernel
:@2
dense/bias
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
(__inference_dense_1_layer_call_fn_366925�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_366936�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
!:	�
@2dense_1/kernel
:@2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_2_layer_call_fn_366942�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_366949�
���
FullArgSpec
args�
jself
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
annotations� *
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
(__inference_dense_2_layer_call_fn_366958�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_366969�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
!:	�@2dense_2/kernel
:@2dense_2/bias
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
(__inference_dropout_layer_call_fn_366974
(__inference_dropout_layer_call_fn_366979�
���
FullArgSpec)
args!�
jself
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_layer_call_and_return_conditional_losses_366984
C__inference_dropout_layer_call_and_return_conditional_losses_366996�
���
FullArgSpec)
args!�
jself
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
annotations� *
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
(__inference_dense_3_layer_call_fn_367005�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_367016�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
 :@2dense_3/kernel
:2dense_3/bias
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
'__inference_output_layer_call_fn_367025�
���
FullArgSpec
args�
jself
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
annotations� *
 z�trace_0
�
�trace_02�
B__inference_output_layer_call_and_return_conditional_losses_367036�
���
FullArgSpec
args�
jself
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
annotations� *
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
&__inference_model_layer_call_fn_366004input_1input_2input_3input_4input_5input_6"�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 
�B�
&__inference_model_layer_call_fn_366471inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 
�B�
&__inference_model_layer_call_fn_366505inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 
�B�
&__inference_model_layer_call_fn_366299input_1input_2input_3input_4input_5input_6"�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_366614inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_366730inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_366349input_1input_2input_3input_4input_5input_6"�
���
FullArgSpec1
args)�&
jself
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
annotations� *
 
�B�
A__inference_model_layer_call_and_return_conditional_losses_366399input_1input_2input_3input_4input_5input_6"�
���
FullArgSpec1
args)�&
jself
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
annotations� *
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
FullArgSpec2
args*�'
jself

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
annotations� *
 0
�B�
$__inference_signature_wrapper_366437input_1input_2input_3input_4input_5input_6"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
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
,__inference_Conv_CDR3_1_layer_call_fn_366739inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_366755inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
2__inference_MaxPooling_CDR3_1_layer_call_fn_366760inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_366768inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
-__inference_Flatten_CDR3_layer_call_fn_366773inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_366779inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
*__inference_reshape_4_layer_call_fn_366784inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
E__inference_reshape_4_layer_call_and_return_conditional_losses_366796inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
,__inference_concatenate_layer_call_fn_366802inputs_0inputs_1"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
G__inference_concatenate_layer_call_and_return_conditional_losses_366809inputs_0inputs_1"�
���
FullArgSpec
args�
jself
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
annotations� *
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
*__inference_reshape_1_layer_call_fn_366814inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
E__inference_reshape_1_layer_call_and_return_conditional_losses_366826inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
*__inference_reshape_2_layer_call_fn_366831inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
E__inference_reshape_2_layer_call_and_return_conditional_losses_366843inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
*__inference_reshape_3_layer_call_fn_366848inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
E__inference_reshape_3_layer_call_and_return_conditional_losses_366860inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
(__inference_reshape_layer_call_fn_366865inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
C__inference_reshape_layer_call_and_return_conditional_losses_366877inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
�B�
.__inference_concatenate_1_layer_call_fn_366886inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
I__inference_concatenate_1_layer_call_and_return_conditional_losses_366896inputs_0inputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec
args�
jself
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
annotations� *
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
&__inference_dense_layer_call_fn_366905inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_366916inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
(__inference_dense_1_layer_call_fn_366925inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_366936inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
.__inference_concatenate_2_layer_call_fn_366942inputs_0inputs_1"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_366949inputs_0inputs_1"�
���
FullArgSpec
args�
jself
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
annotations� *
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
(__inference_dense_2_layer_call_fn_366958inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_366969inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
(__inference_dropout_layer_call_fn_366974inputs"�
���
FullArgSpec)
args!�
jself
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
annotations� *
 
�B�
(__inference_dropout_layer_call_fn_366979inputs"�
���
FullArgSpec)
args!�
jself
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
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_366984inputs"�
���
FullArgSpec)
args!�
jself
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
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_366996inputs"�
���
FullArgSpec)
args!�
jself
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
annotations� *
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
(__inference_dense_3_layer_call_fn_367005inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_367016inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
'__inference_output_layer_call_fn_367025inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
B__inference_output_layer_call_and_return_conditional_losses_367036inputs"�
���
FullArgSpec
args�
jself
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
annotations� *
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
-:+2Adam/m/Conv_CDR3_1/kernel
-:+2Adam/v/Conv_CDR3_1/kernel
#:!2Adam/m/Conv_CDR3_1/bias
#:!2Adam/v/Conv_CDR3_1/bias
$:"	�@2Adam/m/dense/kernel
$:"	�@2Adam/v/dense/kernel
:@2Adam/m/dense/bias
:@2Adam/v/dense/bias
&:$	�
@2Adam/m/dense_1/kernel
&:$	�
@2Adam/v/dense_1/kernel
:@2Adam/m/dense_1/bias
:@2Adam/v/dense_1/bias
&:$	�@2Adam/m/dense_2/kernel
&:$	�@2Adam/v/dense_2/kernel
:@2Adam/m/dense_2/bias
:@2Adam/v/dense_2/bias
%:#@2Adam/m/dense_3/kernel
%:#@2Adam/v/dense_3/kernel
:2Adam/m/dense_3/bias
:2Adam/v/dense_3/bias
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
G__inference_Conv_CDR3_1_layer_call_and_return_conditional_losses_366755k'(3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
,__inference_Conv_CDR3_1_layer_call_fn_366739`'(3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
H__inference_Flatten_CDR3_layer_call_and_return_conditional_losses_366779d3�0
)�&
$�!
inputs���������
� "-�*
#� 
tensor_0����������
� �
-__inference_Flatten_CDR3_layer_call_fn_366773Y3�0
)�&
$�!
inputs���������
� ""�
unknown�����������
M__inference_MaxPooling_CDR3_1_layer_call_and_return_conditional_losses_366768�E�B
;�8
6�3
inputs'���������������������������
� "B�?
8�5
tensor_0'���������������������������
� �
2__inference_MaxPooling_CDR3_1_layer_call_fn_366760�E�B
;�8
6�3
inputs'���������������������������
� "7�4
unknown'����������������������������
!__inference__wrapped_model_365721�'(fgno|}�������
���
���
%�"
input_1���������-
%�"
input_2���������
!�
input_3���������
%�"
input_4���������
%�"
input_5���������

%�"
input_6���������
� "/�,
*
output �
output����������
I__inference_concatenate_1_layer_call_and_return_conditional_losses_366896����
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
.__inference_concatenate_1_layer_call_fn_366886����
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
I__inference_concatenate_2_layer_call_and_return_conditional_losses_366949�Z�W
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
.__inference_concatenate_2_layer_call_fn_366942�Z�W
P�M
K�H
"�
inputs_0���������@
"�
inputs_1���������@
� ""�
unknown�����������
G__inference_concatenate_layer_call_and_return_conditional_losses_366809�\�Y
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
,__inference_concatenate_layer_call_fn_366802�\�Y
R�O
M�J
#� 
inputs_0����������
#� 
inputs_1����������
� ""�
unknown�����������
C__inference_dense_1_layer_call_and_return_conditional_losses_366936dno0�-
&�#
!�
inputs����������

� ",�)
"�
tensor_0���������@
� �
(__inference_dense_1_layer_call_fn_366925Yno0�-
&�#
!�
inputs����������

� "!�
unknown���������@�
C__inference_dense_2_layer_call_and_return_conditional_losses_366969d|}0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_2_layer_call_fn_366958Y|}0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
C__inference_dense_3_layer_call_and_return_conditional_losses_367016e��/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
(__inference_dense_3_layer_call_fn_367005Z��/�,
%�"
 �
inputs���������@
� "!�
unknown����������
A__inference_dense_layer_call_and_return_conditional_losses_366916dfg0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
&__inference_dense_layer_call_fn_366905Yfg0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
C__inference_dropout_layer_call_and_return_conditional_losses_366984c3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
C__inference_dropout_layer_call_and_return_conditional_losses_366996c3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
(__inference_dropout_layer_call_fn_366974X3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@�
(__inference_dropout_layer_call_fn_366979X3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
A__inference_model_layer_call_and_return_conditional_losses_366349�'(fgno|}�������
���
���
%�"
input_1���������-
%�"
input_2���������
!�
input_3���������
%�"
input_4���������
%�"
input_5���������

%�"
input_6���������
p 

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_366399�'(fgno|}�������
���
���
%�"
input_1���������-
%�"
input_2���������
!�
input_3���������
%�"
input_4���������
%�"
input_5���������

%�"
input_6���������
p

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_366614�'(fgno|}�������
���
���
&�#
inputs_0���������-
&�#
inputs_1���������
"�
inputs_2���������
&�#
inputs_3���������
&�#
inputs_4���������

&�#
inputs_5���������
p 

 
� ",�)
"�
tensor_0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_366730�'(fgno|}�������
���
���
&�#
inputs_0���������-
&�#
inputs_1���������
"�
inputs_2���������
&�#
inputs_3���������
&�#
inputs_4���������

&�#
inputs_5���������
p

 
� ",�)
"�
tensor_0���������
� �
&__inference_model_layer_call_fn_366004�'(fgno|}�������
���
���
%�"
input_1���������-
%�"
input_2���������
!�
input_3���������
%�"
input_4���������
%�"
input_5���������

%�"
input_6���������
p 

 
� "!�
unknown����������
&__inference_model_layer_call_fn_366299�'(fgno|}�������
���
���
%�"
input_1���������-
%�"
input_2���������
!�
input_3���������
%�"
input_4���������
%�"
input_5���������

%�"
input_6���������
p

 
� "!�
unknown����������
&__inference_model_layer_call_fn_366471�'(fgno|}�������
���
���
&�#
inputs_0���������-
&�#
inputs_1���������
"�
inputs_2���������
&�#
inputs_3���������
&�#
inputs_4���������

&�#
inputs_5���������
p 

 
� "!�
unknown����������
&__inference_model_layer_call_fn_366505�'(fgno|}�������
���
���
&�#
inputs_0���������-
&�#
inputs_1���������
"�
inputs_2���������
&�#
inputs_3���������
&�#
inputs_4���������

&�#
inputs_5���������
p

 
� "!�
unknown����������
B__inference_output_layer_call_and_return_conditional_losses_367036e��/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
'__inference_output_layer_call_fn_367025Z��/�,
%�"
 �
inputs���������
� "!�
unknown����������
E__inference_reshape_1_layer_call_and_return_conditional_losses_366826d3�0
)�&
$�!
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_reshape_1_layer_call_fn_366814Y3�0
)�&
$�!
inputs���������
� ""�
unknown�����������
E__inference_reshape_2_layer_call_and_return_conditional_losses_366843d3�0
)�&
$�!
inputs���������

� "-�*
#� 
tensor_0����������
� �
*__inference_reshape_2_layer_call_fn_366831Y3�0
)�&
$�!
inputs���������

� ""�
unknown�����������
E__inference_reshape_3_layer_call_and_return_conditional_losses_366860c3�0
)�&
$�!
inputs���������
� ",�)
"�
tensor_0���������~
� �
*__inference_reshape_3_layer_call_fn_366848X3�0
)�&
$�!
inputs���������
� "!�
unknown���������~�
E__inference_reshape_4_layer_call_and_return_conditional_losses_366796d3�0
)�&
$�!
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_reshape_4_layer_call_fn_366784Y3�0
)�&
$�!
inputs���������
� ""�
unknown�����������
C__inference_reshape_layer_call_and_return_conditional_losses_366877d3�0
)�&
$�!
inputs���������-
� "-�*
#� 
tensor_0����������
� �
(__inference_reshape_layer_call_fn_366865Y3�0
)�&
$�!
inputs���������-
� ""�
unknown�����������
$__inference_signature_wrapper_366437�'(fgno|}�������
� 
���
0
input_1%�"
input_1���������-
0
input_2%�"
input_2���������
,
input_3!�
input_3���������
0
input_4%�"
input_4���������
0
input_5%�"
input_5���������

0
input_6%�"
input_6���������"/�,
*
output �
output���������