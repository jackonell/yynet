#include <vector>
/**
 * 激活层，每个节点对于上一层的输入向量x,在与向量w进行内积后，经过激活函数后进行输出
 * 激活函数：可以是sigmoid,relu等函数
 * 一个layer类，包含forward和backprop函数，forward输出，backprop更新w
 * #$#$ 默认实现sigmoid
 */
//template
using namespace std;
class ActivationLayer{
    public:
        virtual ~ActivationLayer(void){}

        void forward(vector<float> nodes_in,vector<float> nodes_out){
            
        }

        void backprop(){

        }
}