from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat.")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando o chat.")
            break

        try:
            answer = chain.invoke(question)
            print(f"RESPOSTA: {answer.strip()}\n")
        except Exception as e:
            print(f"Erro ao processar a pergunta: {e}\n")

        print("---\n")


if __name__ == "__main__":
    main()