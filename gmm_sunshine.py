import numpy as np
from typing import List
import logging
import os
import sys

#配置输出日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]:%(message)s"
)
SOURCE_DATA="data_ps3_ex2.csv"

#读取csv数据
def load_data(src:str,ingnore_head:bool=True,column_select:List[int]=[],encoding="utf-8",sep=",")->np.ndarray:
    '''
    Args:
        src:file path
        ignore_head:whether ignore the first row
        columns_select:list of int,which you want to select
        encoding:file encoding style,default is utf-8
        sep:delimiter,default is ','
    Return:
        data_array:np.ndarray object of int32 dtype

    Raise:
        FileNotFoundError:if src is not exist!
    '''
    if not os.path.exists(src):
        logging.error("%s is not found!"%src)
        sys.exit(0)

    with open(src,mode="r",encoding=encoding) as f:
        if ingnore_head:
            header_names=f.readline().strip().split(sep)
            logging.info("we ignore first head....%s"%str(header_names))
        data_list=[]
        for index,line in enumerate(f,start=1):
            split_line=line.strip().split(sep)
            if column_select:
                item_select=[split_line[item] for item in column_select]
                data_list.append(item_select)
            else:
                data_list.append(split_line)
            if index%20==0:
                logging.info("Reading rows for %d"%index)
    data_array=np.array(
        data_list,
        dtype=np.float32
    )
    return data_array


'''
param theta as follows:
-----------------------
    miu
    theta_x
    theta_y
    though_xy
-----------------------
so p=4,respect that number of params is four

equation is like that:
---------------------------------------------
    x-miu
    y-miu
    (x-miu)**2-theta_x**2
    (y_miu)**2-theta_y**2
    (x-miu)(y-miu)-theta_x*theta_y*though_xy
so q=5,respect that number of equations is five
-----------------------------------------------
'''

#获取统计量,直接推倒公式得出
def get_stat_values(data:np.ndarray):
    """
    获取一些统计值,会在G(theta)中用到,包括x,y的均值,平方和,乘积和,和
    """
    x,y=data[0,],data[1,]
    print(np.std(x),np.std(y))
    x_mean_square=np.mean(
        np.square(x)
    )
    x_mean=np.mean(x)

    y_mean_square=np.mean(
        np.square(y)
    )
    y_mean=np.mean(y)

    xy_mean_multi=np.mean(x*y)
    xy_mean_sum=np.mean(x+y)

    print_info='''stat info of variable x,y
    ------------------------------------------------
    x_mean_square:%.2f
    x_mean:%.2f

    y_mean_square:%.2f
    y_mean:%.2f

    xy_mean_multi:%.2f
    xy_mean_sum:%.2f
    ------------------------------------------------
    '''%(x_mean_square,x_mean,y_mean_square,y_mean,xy_mean_multi,xy_mean_sum)
    logging.info(print_info)
    return [
        x_mean_square,x_mean,
        y_mean_square,y_mean,
        xy_mean_multi,xy_mean_sum
    ]


# data_array=load_data(
#     src=SOURCE_DATA,
#     column_select=[1,2]
# )

# get_stat_values(
#     data=data_array
# )
# print(data_array)

data=np.random.randn(1000,2)

#定义计算过程
def _worker(params,stat_data,W=None):
    """
    Args:
        params:mu,sigma_x,sigma_y,through_xy，给定的theta初始解，用来梯度下降，优化Q value
        stat_data:在计算G(theta)向量时用到的常量,可直接推导公式
        W:方程加权矩阵，qxq矩阵，q是你的方程个数，如果为None,就用单位矩阵替代，用在gmm的第一步估计
    Returns:
        params:通过梯度下降找到的比较优的解
    """
    x_mean_square,x_mean,y_mean_square,y_mean,xy_mean_multi,xy_mean_sum=stat_data
    mu,sigma_x,sigma_y,though_xy=params[0],params[1],params[2],params[3]
    G_param=np.array(
        [
            x_mean-mu,
            y_mean-mu,
            x_mean_square-2*x_mean*mu+mu**2-sigma_x**2,
            y_mean_square-2*y_mean*mu+mu**2-sigma_y**2,
            xy_mean_multi-xy_mean_sum*mu+mu**2-though_xy*sigma_x*sigma_y
        ]
    ).reshape(-1,1)

    G_param_T=np.transpose(G_param)

    #方程的个数
    q_number=G_param.shape[0]
    #参数的个数
    p_number=len(params)

    #定义梯度矩阵,参数个数=4,方程个数=5 so the shape is (5,4)
    G_grad=np.array(
        [
            [-1,0,0,0],
            [-1,0,0,0],
            [2*(mu-x_mean),-2*sigma_x,0,0],
            [2*(mu-y_mean),0,-2*sigma_y,0],
            [2*(mu-xy_mean_sum),-sigma_x*though_xy,-sigma_y*though_xy,-sigma_x*sigma_y]
        ]
    )

    #获取其转置矩阵
    G_grad_T=np.transpose(G_grad)
    
    if W is None:
    #使用W为初始单位方阵
        W=np.eye(q_number,dtype=np.float32)

    #计算Q值
    Q_params=np.dot(
        np.dot(G_param_T,W),G_param
    )
    
    #计算Q关于 参数的梯度
    Q_grad=2*np.dot(
        np.dot(G_grad_T,W),G_param
    )

    return Q_params,Q_grad


def gmm_first_step(lr=0.01,verbose=True):
    """
    lr:梯度下降的学习率，就是步长
    verbose:是否显示迭代过程,默认显示
    """
    # data=load_data(
    #     src=SOURCE_DATA,
    #     column_select=[1,2]
    # )

    # x_mean_square,x_mean,y_mean_square,y_mean,xy_mean_multi,xy_mean_sum=get_stat_values(data)
    stat_data=get_stat_values(data)
    params={
        "mu":0.1,
        "sigma_x":1.2,
        "sigma_y":1.5,
        "though_xy":0
    }

    mu=params["mu"]
    sigma_x=params["sigma_x"]
    sigma_y=params["sigma_y"]
    though_xy=params["though_xy"]

    init_params=np.array([mu,sigma_x,sigma_y,though_xy])
    #定义G矩阵
    _params=init_params
    
    logging.info("beging evalute theta by frist GMM FIRST STEP....")
    for step in range(30):
        Q_params,Q_grad=_worker(_params,stat_data)
        logging.info("iter at %d,the Q value is %.4f"%(step,Q_params[0,0]))
        _params=_params-lr*Q_grad.reshape(-1,)
        # print(_params)
    return _params

def evaliuate_W_by_first_step(_params):
    """
    根据 gmm first step的结果计算W
    Args:   
        _param:gmm first求的theta
    Return:
        W:加权矩阵
    """
    #unwarp param
    mu,sigma_x,sigma_y,though_xy=_params
    def _calc(x,y):
        g_param=np.array(
                [
                    x-mu,
                    y-mu,
                    (x-mu)**2-sigma_x**2,
                    (y-mu)**2-sigma_y**2,
                    (x-mu)*(y-mu)-sigma_x*sigma_y*though_xy
                ]
            ).reshape(-1,1)
        g_param_T=np.transpose(g_param)
        result=np.dot(g_param,g_param_T)
        return result
    
    rows,_=data.shape
    W=np.zeros(shape=(5,5))
    #严格按照公式，先求矩阵乘积
    for i in range(rows):
        x,y=data[i,]
        W=W+_calc(x, y)

    #再求其逆矩阵
    W=np.linalg.inv(W)
    return W

def gmm_second_step(params,W,lr=0.01,verbose=True):
    """
    根据上一步求的W求解theta
    参数含义同上
    """
    logging.info("begin evalue theta by GMM SECOND STEP...")
    stat_data=get_stat_values(data)
    lr=0.1
    for step in range(30):
        Q_params,Q_grad=_worker(params,stat_data,W=W) #指定第一次估计的W
        logging.info("iter at %d,the Q value is %.4f"%(step,Q_params[0,0]))
        params=params-lr*Q_grad.reshape(-1,)
    return params
    

def gmm_theta_var(params,W):
    """
    求解theta 的方差，严格按照公式
    Args:
        params:theta
        W:加权矩阵,B的逆矩阵就是W
    """
    x_mean_square,x_mean,y_mean_square,y_mean,xy_mean_multi,xy_mean_sum=get_stat_values(data)
    mu,sigma_x,sigma_y,though_xy=params[0],params[1],params[2],params[3]
    G_param=np.array(
        [
            x_mean-mu,
            y_mean-mu,
            x_mean_square-2*x_mean*mu+mu**2-sigma_x**2,
            y_mean_square-2*y_mean*mu+mu**2-sigma_y**2,
            xy_mean_multi-xy_mean_sum*mu+mu**2-though_xy*sigma_x*sigma_y
        ]
    ).reshape(-1,1)

    G_grad=np.array(
        [
            [-1,0,0,0],
            [-1,0,0,0],
            [2*(mu-x_mean),-2*sigma_x,0,0],
            [2*(mu-y_mean),0,-2*sigma_y,0],
            [2*(mu-xy_mean_sum),-sigma_x*though_xy,-sigma_y*though_xy,-sigma_x*sigma_y]
        ]
    )

    G_grad_transpose=np.transpose(G_grad)
    #np.dot 这个api是求解矩阵叉乘
    theta_variance=np.mean(
        np.dot(
            np.dot(G_grad_transpose,W),G_grad
        )
    )
    return theta_variance

def J_test(W,theta):
    """
    我不晓得公式
    """
    rows,_=data.shape
    mu,sigma_x,sigma_y,though_xy=theta
    x_mean_square,x_mean,y_mean_square,y_mean,xy_mean_multi,xy_mean_sum=get_stat_values(data)
    G_param=np.array(
        [
            x_mean-mu,
            y_mean-mu,
            x_mean_square-2*x_mean*mu+mu**2-sigma_x**2,
            y_mean_square-2*y_mean*mu+mu**2-sigma_y**2,
            xy_mean_multi-xy_mean_sum*mu+mu**2-though_xy*sigma_x*sigma_y
        ]
    ).reshape(-1,1)
    G_param_T=np.transpose(G_param)
    J=rows*np.dot(
        np.dot(G_param_T,W),G_param
    )
    return J

first_step_params=gmm_first_step()

W=evaliuate_W_by_first_step(first_step_params)

second_params=gmm_second_step(
    params=first_step_params,
    W=W
)
print(second_params)
theta_variance=gmm_theta_var(second_params, W)

# print(second_params)
print("Theta VARIANCE is %.6f"%theta_variance)

J=J_test(W,second_params)
print("J stat is %.4f"%J)

