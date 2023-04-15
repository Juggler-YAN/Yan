class BinStrTree{
public:
    BinStrTree() : root(new TreeNode()) {}
    BinStrTree(const BinStrTree &bst) : root(new TreeNode(*bst.root)) {}
    BinStrTree& operator=(const BinStrTree &);
    ~BinStrTree() { delete root; }
private:
	TreeNode *root;	
};

BinStrTree& BinStrTree::operator=(const BinStrTree &bst) {
    auto newroot = new TreeNode(*bst.root);
    delete root;
    root = newroot;
    return *this;
}