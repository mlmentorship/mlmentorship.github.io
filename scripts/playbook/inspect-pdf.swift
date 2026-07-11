#!/usr/bin/env swift
import Foundation
import PDFKit

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("PDF inspection failed: \(message)\n".utf8))
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: swift scripts/playbook/inspect-pdf.swift <playbook.pdf>")
}

let pdfURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard let document = PDFDocument(url: pdfURL) else {
    fail("could not open \(pdfURL.path)")
}

let pageCount = document.pageCount
guard (8...60).contains(pageCount) else {
    fail("unexpected page count: \(pageCount)")
}

var totalCharacters = 0
var sparsePages: [Int] = []
var urlAnnotations = 0
var pageSummaries: [[String: Any]] = []

for index in 0..<pageCount {
    guard let page = document.page(at: index) else {
        fail("missing page \(index + 1)")
    }
    let text = (page.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    let characters = text.count
    totalCharacters += characters
    if characters < 80 { sparsePages.append(index + 1) }

    for annotation in page.annotations {
        if annotation.action is PDFActionURL || annotation.url != nil {
            urlAnnotations += 1
        }
    }

    let bounds = page.bounds(for: .mediaBox)
    pageSummaries.append([
        "page": index + 1,
        "characters": characters,
        "width": Int(bounds.width.rounded()),
        "height": Int(bounds.height.rounded()),
    ])
}

guard totalCharacters > 3_000 else {
    fail("too little extractable text: \(totalCharacters) characters")
}
guard sparsePages.isEmpty else {
    fail("near-empty page(s): \(sparsePages.map(String.init).joined(separator: ", "))")
}
guard urlAnnotations > 0 else {
    fail("no clickable resource links found")
}

let result: [String: Any] = [
    "file": pdfURL.lastPathComponent,
    "pages": pageCount,
    "characters": totalCharacters,
    "clickableLinks": urlAnnotations,
    "pageSummaries": pageSummaries,
]
let json = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(json)
FileHandle.standardOutput.write(Data("\n".utf8))
