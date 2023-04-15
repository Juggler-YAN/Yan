#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>
#include <set>

using namespace std;

class Folder;

class Message {
    friend class Folder;
    friend void swap(Message &, Message &);
public:
    explicit Message(const string &str = "") : contents(str) {}
    Message(const Message&);
    Message(Message &&);
    Message& operator=(const Message&);
    Message& operator=(Message&&);
    ~Message();
    void save(Folder&);
    void remove(Folder&);
    void move_Folders(Message*);
private:
    string contents;
    set<Folder*> folders;
    void add_to_Folders(const Message&);
    void remove_from_Folders();
	void addFldr(Folder *f) { folders.insert(f); }
	void remFldr(Folder *f) { folders.erase(f); }
};

class Folder {
    friend class Message;
    friend void swap(Message &, Message &);
public:
    Folder() = default;
    Folder(const Folder&);
    Folder& operator=(const Folder&);
    ~Folder();
private:
    set<Message*> messages;
    void add_to_Messages(const Folder&);
    void remove_from_Messages();
    void addMsg(Message *m) { messages.insert(m); };
    void remMsg(Message *m) { messages.erase(m); };
};

void Message::add_to_Folders(const Message &m) {
    for (auto f : m.folders) {
        f->addMsg(this);
    }
}

void Message::remove_from_Folders() {
    for (auto f : folders) {
        f->remMsg(this);
    }
}

void Message::save(Folder &f) {
    folders.insert(&f);
    f.addMsg(this);
}

void Message::remove(Folder &f) {
    folders.erase(&f);
    f.remMsg(this);
}

void Message::move_Folders(Message *m) {
    folders = std::move(m->folders);
    for (auto f : folders) {
        f->remMsg(m);
        f->addMsg(this);
    }
    m->folders.clear();
}

Message::Message(const Message &m) : contents(m.contents), folders(m.folders) {
    add_to_Folders(m);
}

Message::Message(Message &&m) : contents(std::move(m.contents)) {
    move_Folders(&m);
}

Message& Message::operator=(const Message &rhs) {
    remove_from_Folders();
    contents = rhs.contents;
    folders = rhs.folders;
    add_to_Folders(rhs);
    return *this;
}

Message& Message::operator=(Message &&rhs) {
    if (this != &rhs) {
        remove_from_Folders();
        contents = std::move(rhs.contents);
        move_Folders(&rhs);
    }
    return *this;
}

Message::~Message() {
    remove_from_Folders();
}

void Folder::add_to_Messages(const Folder &f) {
    for (auto m : f.messages) {
        m->addFldr(this);
    }
}

void Folder::remove_from_Messages() {
    for (auto m : messages) {
        m->remFldr(this);
    }
}

Folder::Folder(const Folder& f) : messages(f.messages) {
    add_to_Messages(f);
}

Folder& Folder::operator=(const Folder& rhs) {
    remove_from_Messages();
    messages = rhs.messages;
    add_to_Messages(rhs);
    return *this;
}

Folder::~Folder() {
    remove_from_Messages();
}

void swap(Message &lhs, Message &rhs) {
    using std::swap;
    for (auto f : lhs.folders) {
        f->remMsg(&lhs);
    }
    for (auto f : rhs.folders) {
        f->remMsg(&rhs);
    }
    swap(lhs.contents, rhs.contents);
    swap(lhs.folders, rhs.folders);
    for (auto f : lhs.folders) {
        f->addMsg(&lhs);
    }
    for (auto f : rhs.folders) {
        f->addMsg(&rhs);
    }
}

#endif