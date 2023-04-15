#include <string>

using namespace std;

class TreeNode {
public:
    TreeNode() : value(""), count(0), left(nullptr), right(nullptr) {}
    TreeNode(const TreeNode &tn) : value(tn.value), count(tn.count), left(tn.left), right(tn.right) { ++*count; }
    TreeNode& operator=(const TreeNode &);
    ~TreeNode();
private:
	string value;
	int *count;
	TreeNode *left;
	TreeNode *right;	
};
TreeNode& TreeNode::operator=(const TreeNode &tn) {
    ++*tn.count;
    if (--*count == 0) {
        if (left) {
            delete left;
        }
        if (right) {
            delete right;
        }
        delete count;
    }
    value = tn.value;
    count = tn.count;
    left = tn.left;
    right = tn.right;
    return *this;
}
TreeNode::~TreeNode() {
    if (--*count == 0) {
        if (left) {
            delete left;
        }
        if (right) {
            delete right;
        }
        delete count;
    }
}