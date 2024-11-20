#include "tree.h"
#include <algorithm>
#include <sstream>

std::string PathUtils::getParentPath(const std::string& path) {
    if (path.empty() || path == "/") return "/";
    size_t pos = path.find_last_of('/');
    if (pos == 0) return "/";
    if (pos == std::string::npos) return "/";
    return path.substr(0, pos);
}

std::string PathUtils::getLastSegment(const std::string& path) {
    if (path.empty() || path == "/") return "";
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

std::string PathUtils::joinPaths(const std::string& path1, const std::string& path2) {
    std::string result = path1;
    if (!result.empty() && result.back() != '/') result += '/';
    if (!path2.empty() && path2.front() == '/') return result + path2.substr(1);
    return result + path2;
}

bool PathUtils::isAbsPath(const std::string& path) {
    return !path.empty() && path[0] == '/';
}

std::vector<std::string> PathUtils::parse(const std::string& path) {
    std::vector<std::string> result;
    std::stringstream ss(path);
    std::string segment;
    while (std::getline(ss, segment, '/')) {
        if (!segment.empty()) {
            result.push_back(segment);
        }
    }
    return result;
}

std::string PathUtils::parseRelativePath(const std::string& path, const std::string& currentPath) {
    if (path.empty()) return currentPath;
    if (isAbsPath(path)) return normalize(path);
    if (!isAbsPath(currentPath)) {
        throw std::invalid_argument("current_path must be an absolute path");
    }

    std::string workPath = path;
    if (workPath.substr(0, 2) == "./") {
        workPath = workPath.substr(2);
    }

    std::string fullPath = currentPath;
    if (fullPath.back() != '/') fullPath += '/';
    fullPath += workPath;

    return normalize(fullPath);
}

std::string PathUtils::combine(const std::vector<std::string>& parts) {
    if (parts.empty()) return "/";
    std::string result;
    for (const auto& part : parts) {
        result += "/" + part;
    }
    return result;
}

std::string PathUtils::normalize(const std::string& path) {
    std::vector<std::string> parts = parse(path);
    std::vector<std::string> result;
    
    for (const auto& part : parts) {
        if (part == "..") {
            if (!result.empty()) {
                result.pop_back();
            }
        }
        else if (part != ".") {
            result.push_back(part);
        }
    }
    
    return combine(result);
}

std::string PathUtils::getRelativePath(const std::string& targetPath, const std::string& relPath) {
    auto relParts = parse(relPath);
    auto targetParts = parse(targetPath);

    size_t commonPrefixLen = 0;
    size_t minLen = std::min(relParts.size(), targetParts.size());
    
    for (size_t i = 0; i < minLen; ++i) {
        if (relParts[i] != targetParts[i]) break;
        commonPrefixLen = i + 1;
    }

    size_t upLevels = relParts.size() - commonPrefixLen;
    std::vector<std::string> relativeParts(upLevels, "..");
    
    for (size_t i = commonPrefixLen; i < targetParts.size(); ++i) {
        relativeParts.push_back(targetParts[i]);
    }

    if (relativeParts.empty()) return ".";
    
    std::string result;
    for (size_t i = 0; i < relativeParts.size(); ++i) {
        if (i > 0) result += "/";
        result += relativeParts[i];
    }
    
    return result;
}

