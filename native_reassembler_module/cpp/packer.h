#ifndef IMAGE_REASSEMBLER_PACKER_H
#define IMAGE_REASSEMBLER_PACKER_H

#include "Rect.h"
#include <vector>

class Block {
public:
    int x, y, w, h;
    bool used;
    Block *down;
    Block *right;

    Block(int x_, int y_, int w_, int h_)
            : x(x_), y(y_), w(w_), h(h_), used(false), down(nullptr), right(nullptr) {}

    ~Block() {
        delete down;
        delete right;
    }
};

class Packer {
public:
    Packer(int w, int h);

    std::vector<RectMapping> fit(const std::vector<Rect> &rects);

private:
    Block *find_node(Block *n, int w, int h);

    Block *split_node(Block *node, int w, int h);

    Block root;
};

class ResizablePacker {
public:
    ResizablePacker();

    ~ResizablePacker() { delete root; }

    std::vector<RectMapping> fit(const std::vector<Rect> &rects);

private:
    Block *find_node(Block *n, int w, int h);

    Block *split_node(Block *node, int w, int h);

    Block *grow_node(int w, int h);

    Block *grow_right(int w, int h);

    Block *grow_down(int w, int h);

    Block *root;
};

#endif //IMAGE_REASSEMBLER_PACKER_H
