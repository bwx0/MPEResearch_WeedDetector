#include "packer.h"

Packer::Packer(int w, int h) : root(0, 0, w, h) {}

std::vector<RectMapping> Packer::fit(const std::vector<Rect>& rects) {
    std::vector<RectMapping> result;
    for (const Rect& rect : rects) {
        Block* node = find_node(&root, rect.w, rect.h);
        if (node) {
            Block* fit = split_node(node, rect.w, rect.h);
            Rect fit_rect(fit->x, fit->y, rect.w, rect.h);
            result.emplace_back(rect, fit_rect);
        } else {
            result.emplace_back(rect, std::nullopt);
        }
    }
    return result;
}

Block* Packer::find_node(Block* n, int w, int h) {
    if (n->used) {
        Block* node = find_node(n->right, w, h);
        if (node)
            return node;
        return find_node(n->down, w, h);
    } else if (w <= n->w && h <= n->h) {
        return n;
    } else {
        return nullptr;
    }
}

Block* Packer::split_node(Block* node, int w, int h) {
    node->used = true;
    node->down = new Block(node->x, node->y + h, node->w, node->h - h);
    node->right = new Block(node->x + w, node->y, node->w - w, h);
    return node;
}

ResizablePacker::ResizablePacker() : root(nullptr) {}

std::vector<RectMapping> ResizablePacker::fit(const std::vector<Rect>& rects) {
    if (rects.empty())
        return {};

    int w = rects[0].w;
    int h = rects[0].h;
    root = new Block(0, 0, w, h);

    std::vector<RectMapping> result;
    for (const Rect& rect : rects) {
        Block* node = find_node(root, rect.w, rect.h);
        std::optional<Rect> fit_rect = std::nullopt;
        if (node) {
            Block* fit = split_node(node, rect.w, rect.h);
            fit_rect = Rect(fit->x, fit->y, rect.w, rect.h);
        } else {
            Block* fit = grow_node(rect.w, rect.h);
            if (fit) {
                fit_rect = Rect(fit->x, fit->y, rect.w, rect.h);
            }
        }
        result.emplace_back(rect, fit_rect);
    }
    return result;
}

Block* ResizablePacker::find_node(Block* n, int w, int h) {
    if (n->used) {
        Block* node = find_node(n->right, w, h);
        if (node)
            return node;
        return find_node(n->down, w, h);
    } else if (w <= n->w && h <= n->h) {
        return n;
    } else {
        return nullptr;
    }
}

Block* ResizablePacker::split_node(Block* node, int w, int h) {
    node->used = true;
    node->down = new Block(node->x, node->y + h, node->w, node->h - h);
    node->right = new Block(node->x + w, node->y, node->w - w, h);
    return node;
}

Block* ResizablePacker::grow_node(int w, int h) {
    bool can_grow_down = w <= root->w;
    bool can_grow_right = h <= root->h;

    bool should_grow_right = can_grow_right && (root->h >= root->w + w);
    bool should_grow_down = can_grow_down && (root->w >= root->h + h);

    if (should_grow_right)
        return grow_right(w, h);
    else if (should_grow_down)
        return grow_down(w, h);
    else if (can_grow_right)
        return grow_right(w, h);
    else if (can_grow_down)
        return grow_down(w, h);
    else
        return nullptr;
}

Block* ResizablePacker::grow_right(int w, int h) {
    Block* old_root = root;
    root = new Block(0, 0, old_root->w + w, old_root->h);
    root->used = true;
    root->down = old_root;
    root->right = new Block(old_root->w, 0, w, old_root->h);
    Block* node = find_node(root, w, h);
    if (node)
        return split_node(node, w, h);
    else
        return nullptr;
}

Block* ResizablePacker::grow_down(int w, int h) {
    Block* old_root = root;
    root = new Block(0, 0, old_root->w, old_root->h + h);
    root->used = true;
    root->down = new Block(0, old_root->h, old_root->w, h);
    root->right = old_root;
    Block* node = find_node(root, w, h);
    if (node)
        return split_node(node, w, h);
    else
        return nullptr;
}

